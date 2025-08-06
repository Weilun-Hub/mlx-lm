# Copyright © 2023-2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope

import numpy as np

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    dim_model_base: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    scale_depth: float
    scale_emb: float
    max_position_embeddings: Optional[int] = None
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[str, float]]] = None
    tie_word_embeddings: bool = False
    sparse_config: Optional[Dict[str, Any]] = None


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1] # [8192]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = mx.max(batch_sizes).item() # 8192
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1 # (8192 - 32) // 16 + 1 = 511
    chunk_start_offsets = mx.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride) # [0, 16, 32, ..., 8144, 8160], shape = 511
    seq_starts = cu_seqlen[:-1] # [0]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq], shape = (1, 511)

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size # [0 + 32, 16 + 32, 32 + 32, ..., 8144 + 32, 8160 + 32] = [32, 48, 64, ..., 8176, 8192], shape = (1, 511)
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])) # shape = (1, 511), [True, True, True, ..., True, True]

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_mask_npy = np.array(valid_chunk_mask)
    chunk_start_in_seq_npy = np.array(chunk_start_in_seq)
    valid_chunk_starts_npy = chunk_start_in_seq_npy[valid_chunk_mask_npy]  # [num_valid_chunks], shape = 874, [0, 16, 32, ..., 8144, 8160]
    valid_chunk_starts = mx.array(valid_chunk_starts_npy)
    del chunk_start_in_seq, chunk_start_in_seq_npy
    # 5. Generate filtered_indices
    chunk_indices = mx.arange(0, chunk_size)[None, :]  # [1, chunk_size], [0, 1, 2, ..., 31]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size], shape = (511, 32)
    filtered_indices = filtered_indices.reshape(-1)  # Flatten to 1D indices [0, 1, 2, ..., 30, 31, | 16, 17, 46, 47, | 32, 33, ..., 62, 63, ... ]

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = mx.sum(valid_chunk_mask, axis=1) # [511]
    cu_seqlens_compressed = mx.zeros(len(cu_seqlen), dtype=mx.int32) # [0, 0]
    # cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0) # [0, 874]
    cu_seqlens_compressed[1:] = mx.cumsum(num_filtered_chunks_per_batch, axis=0) # [0, 511]
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, valid_chunk_mask_npy, chunk_indices
    return filtered_indices, cu_seqlens_compressed # (511 * 32), [0, 511]

class CompressK(nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Module for compressing key (K) representations.

        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size # 32
        self.head_num_k = head_num_k # 2
        self.head_dim = head_dim # 128
        self.kernel_stride = kernel_stride # 16

    def __call__(
        self, 
        k: mx.array,
        cu_seqlens: mx.array
    ) -> Tuple[mx.array, mx.array]:
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        ) # (511 * 32), [0, 511]
        B, H, L, D = k.shape # 1, 2, 8192, 128
        k_ = mx.transpose(k, (2, 0, 1, 3)) # 8192, 1, 2, 128
        filtered_k_ = mx.gather(k_, filtered_k_indices, axis=0, slice_sizes=(1, B, H, D)) # 16352, 1, 1, 2, 128
        filtered_k_ = mx.squeeze(filtered_k_, axis=1) # 16352, 1, 2, 128
        filtered_k = mx.transpose(filtered_k_, (1, 2, 0, 3)) # 1, 2, 16352, 128
        filtered_k = filtered_k.reshape(B, H, filtered_k.shape[2] // self.kernel_size, self.kernel_size, D) # 1, 2, 511, 32, 128
        compressed_k = filtered_k.mean(axis=3) # 1, 2, 511, 128
        return compressed_k, cu_seqlens_compressed

def compressed_attention(
    queries,
    compressed_keys,
    compressed_values,
    cache,
    kernel_size,
    kernel_stride,
    block_size,
    topk,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    init_blocks,
    local_blocks,
    total_seq_lens,
    scale
):
    batch_size = cu_seqlens_q.shape[0] - 1 # 2 - 1 = 1

    cache_len = total_seq_lens - max_seqlen_q
    assert batch_size == 1, "Batch size must be 1 for current implementation, but got {}".format(batch_size)
    # q_idx = mx.array([total_seq_lens - 1], dtype=mx.int32) // block_size
    
    _, _, qL, D = queries.shape # 1, 32, 2048, 128
    
    # scale = 1. / float(mx.sqrt(D))
    mask = None if qL == 1 else "causal"

    score = mx.fast.infllmv2_attention_stage1(
        mx.contiguous(queries), # 1, 32, 2048, 128
        mx.contiguous(compressed_keys), # 1, 2, 511, 128
        mx.contiguous(compressed_values), # 1, 2, 511, 128
        scale=scale, # 1 / sqrt(128)
        mask=mask # "causal"
    )
    
    stride = block_size // kernel_stride
    kernel_size = stride + 1
    padding = 1
    cache_len = cache.offset - qL
    pooled_score = mx.maxpooling(
        score, # (1, 2, 2048, 511)
        cache_len, # 8192 - 2048 = 6144
        init_blocks, # 1
        local_blocks, # 32
        kernel_size, # 5
        stride, # 4
        padding, # 1
        block_size # 64
    ) # (1, 2, 2048, 128)

    topk = min(topk, pooled_score.shape[3])
    topk_idx = mx.argtopk(pooled_score, topk, axis=-1) # (1, 2, 2048, 64)
    
    return topk_idx

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.num_heads = n_heads = args.num_attention_heads
        self.rope_theta = args.rope_theta

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

        # sparse attention config
        self.kernel_size = self.args.sparse_config.get("kernel_size", 32) # 32
        self.kernel_stride = self.args.sparse_config.get("kernel_stride", 16) # 16
        self.init_blocks = self.args.sparse_config.get("init_blocks", 1) # 1
        self.block_size = self.args.sparse_config.get("block_size", 64) # 64
        self.window_size = self.args.sparse_config.get("window_size", 2048) # 2048
        self.dense_len = self.args.sparse_config.get("dense_len", 8192) # 8192

        self.local_blocks = self.window_size // self.block_size # 2048 // 64 = 32
        self.topk = self.args.sparse_config.get("topk", 64) # 64
        self.use_nope = self.args.sparse_config.get("use_nope", False) # False

        self.compress_k = CompressK(self.num_key_value_heads, self.head_dim, self.kernel_size, self.kernel_stride)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        B, L, _ = x.shape # 1, 6, 4096. B = 1, L = 2048 for chunked prefill

        assert B == 1, "Batch size must be 1 for current implementation, but got {}".format(B)

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x) # queries: 1, 6, 4096 -> 1, 6, 4096. keys/values: 1, 6, 4096 -> 1, 6, 256.

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3) # 1, 6, 4096 -> 1, 6, 32, 128 -> 1, 32, 6, 128
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3) # 1, 6, 256 -> 1, 6, 2, 128 -> 1, 2, 6, 128
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        ) # 1, 6, 256 -> 1, 6, 2, 128 -> 1, 2, 6, 128

        assert cache is not None, "Cache must be provided for compressed attention"

        queries = self.rope(queries, offset=cache.offset) # 1, 32, 6, 128 -> 1, 32, 6, 129
        keys = self.rope(keys, offset=cache.offset) # 1, 2, 6, 128 -> 1, 2, 6, 128
        keys, values = cache.update_and_fetch(keys, values) # keys / values: 1, 2, 6, 128 -> 1, 2, 6, 128
        
        if cache.offset < self.dense_len:
            attn_output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            ) # queries: (1, 32, 2048, 128), keys: (1, 2, 4096, 128), values: (1, 2, 4096, 128)
        else:
            # queries: (1, 32, 2048, 128), keys/values: (1, 2, 8192, 128)
            attn_output = self.sparse_forward(queries, keys, values, cache, self.scale, mask)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output)

    def sparse_forward(self, queries, keys, values, cache, scale, mask): # cu_seqlens_q, cu_seqlens_k, max_seqlen_in_batch_q, max_seqlen_in_batch_k,
        
        _, _, qL, _ = queries.shape
        _, _, kL, _ = keys.shape

        new_keys = keys[:, :, kL - qL:, :]

        if cache.compressed_keys is None:
            cu_seqlens_k = mx.array([0, kL], dtype=mx.int32)
            compressed_keys, compressed_cu_seqlens = self.compress_k(keys, cu_seqlens_k)
            
            no_compressed_k_start = compressed_keys.shape[2] * self.kernel_stride # 511 * 16 = 8176
            cache.update_compressed_keys(compressed_keys)
            cache.update_no_compressed_keys(keys[:, :, no_compressed_k_start:, :], no_compressed_k_start)
            cache.cached_compressed_cu_seqlens.append(compressed_cu_seqlens)
        else:
            if qL > 1:
                no_compressed_k = cache.update_no_compressed_keys(new_keys, kernel_size=new_keys.shape[2], kernel_stride=new_keys.shape[2])
            else:
                no_compressed_k = cache.update_no_compressed_keys(new_keys, kernel_size=self.kernel_size, kernel_stride=self.kernel_stride)
            
            if no_compressed_k is not None:
                if qL > 1:
                    cu_seqlens_k = mx.array([0, no_compressed_k.shape[2]], dtype=mx.int32)
                    compressed_keys, compressed_cu_seqlens = self.compress_k(no_compressed_k, cu_seqlens_k)
                    compressed_keys = cache.update_compressed_keys(compressed_keys)
                    cache.cached_compressed_cu_seqlens[0][-1] += compressed_cu_seqlens[1] - compressed_cu_seqlens[0]
                    compressed_cu_seqlens = cache.cached_compressed_cu_seqlens[0]
                else:
                    compressed_keys = no_compressed_k.mean(axis=3, keepdims=True)
                    compressed_keys = cache.update_compressed_keys(compressed_keys)
                    cache.cached_compressed_cu_seqlens[0][-1] += 1
                    compressed_cu_seqlens = cache.cached_compressed_cu_seqlens[0]
            else:
                compressed_keys = cache.compressed_keys
                compressed_cu_seqlens = cache.cached_compressed_cu_seqlens[0]
        
        compressed_values = compressed_keys

        cu_seqlens_q = mx.array([0, qL], dtype=mx.int32)
        cu_seqlens_k = mx.array([0, kL], dtype=mx.int32)
        max_seqlen_q = qL
        max_seqlen_k = kL
        
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1] # [511]

        topk_idx = compressed_attention(
            queries,
            compressed_keys,
            compressed_values,
            cache,
            self.kernel_size, # 32
            self.kernel_stride, # 16
            self.block_size, # 64
            self.topk, # 64
            cu_seqlens_q, # [0, 2048]
            compressed_cu_seqlens, # [0, 511]
            max_seqlen_q, # 2048
            compressed_seqlens.max().item(), # 511
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            total_seq_lens=cache.offset,
            scale=scale
        ) # (1, 2, 2048, 64)

        blockmask_uint64 = mx.topk_to_uint64(
            topk_idx, # (1, 2, 2048, 64)
            cache.offset, # 8192
            self.block_size # 64
        ) # (1, 2, 2048, 2)
        mx.eval(blockmask_uint64)

        window_size_left = -1
        window_size_right = -1
        block_window_size = self.window_size // self.block_size # 2048 // 64 = 32

        topk_attn_output = mx.fast.infllmv2_attention_stage2(
            queries, 
            keys, 
            values, 
            cu_seqlens_q, 
            cu_seqlens_k,
            max_seqlen_q, 
            max_seqlen_k, 
            window_size_left, 
            window_size_right, 
            blockmask_uint64, 
            block_window_size, 
            scale=scale, 
            mask=mask
        ) # (1, 32, 2048, 128)

        mx.eval(topk_attn_output)
        # mx.clear_cache()

        return topk_attn_output

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers

        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.scale_depth = args.scale_depth
        self.num_hidden_layers = args.num_hidden_layers

    def __call__(
        self,
        x: mx.array, # 1, 6,4096
        mask: Optional[mx.array] = None, # "causal"
        cache: Optional[Any] = None, # mlx_lm.models.cache.KVCache
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * (self.scale_depth / self.num_hidden_layers**0.5)
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r * (self.scale_depth / self.num_hidden_layers**0.5)
        return out


class MiniCPM4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs) * self.args.scale_emb # self.args.scale_emb = 12, (1, 6, 4096)

        if mask is None:
            mask = create_attention_mask(h, cache) # here, mask = "causal"

        if cache is None: # false
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache): # 0, 1, ..., 31
            h = layer(h, mask, c)

        return self.norm(h) # (1, 2048, 4096)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniCPM4Model(args)

        if not self.args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)

        if not self.args.tie_word_embeddings: # here
            out = self.lm_head(out / (self.args.hidden_size / self.args.dim_model_base)) # 4096 / 256 = 16
        else:
            out = out @ self.model.embed_tokens.weight.T

        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        return weights

    @property
    def layers(self):
        return self.model.layers
