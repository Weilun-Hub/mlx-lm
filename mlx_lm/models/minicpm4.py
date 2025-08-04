# Copyright © 2023-2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


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
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1] # [14001]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = mx.max(batch_sizes) # 14001
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1 # (14001 - 32) // 16 + 1 = 874
    chunk_start_offsets = mx.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device) # [0, 16, 32, ..., 13952, 13968], shape = 874
    seq_starts = cu_seqlen[:-1] # [0]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq], shape = (1, 874)

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size # [0 + 32, 16 + 32, 32 + 32, ..., 13952 + 32, 13968 + 32] = [32, 48, 64, ..., 14000, 14016], shape = (1, 874)
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])) # shape = (1, 874), [True, True, True, ..., True, True]

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks], shape = 874, [0, 16, 32, ..., 13952, 13968]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = mx.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size], [0, 1, 2, ..., 31]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size], shape = (874, 32)
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices [0, 1, 2, ..., 30, 31, | 16, 17, 46, 47, | 32, 33, ..., 62, 63, ... ]

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch, [874]
    cu_seqlens_compressed = mx.zeros(
        len(cu_seqlen), dtype=mx.int32, device=cu_seqlen.device
    ) # [0, 0]
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0) # [0, 874]
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed # (874 * 32), [0, 874]


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
        )
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(filtered_k.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim)  # [l, block_size,h,d]
        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed

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
        B, L, _ = x.shape # 1, 6, 4096. B = 1, L = 4096

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x) # queries: 1, 6, 4096 -> 1, 6, 4096. keys/values: 1, 6, 4096 -> 1, 6, 256.

        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3) # 1, 6, 4096 -> 1, 6, 32, 128 -> 1, 32, 6, 128
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3) # 1, 6, 256 -> 1, 6, 2, 128 -> 1, 2, 6, 128
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        ) # 1, 6, 256 -> 1, 6, 2, 128 -> 1, 2, 6, 128

        if cache is not None: # here
            queries = self.rope(queries, offset=cache.offset) # 1, 32, 6, 128 -> 1, 32, 6, 129
            keys = self.rope(keys, offset=cache.offset) # 1, 2, 6, 128 -> 1, 2, 6, 128
            keys, values = cache.update_and_fetch(keys, values) # keys / values: 1, 2, 6, 128 -> 1, 2, 6, 128
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        if cache.offset < self.dense_len:
            attn_output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        elif cache.keys is None or L != 1: # prefill
            pass
        else:
            pass

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(attn_output)


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

        return self.norm(h)


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
            out = self.lm_head(out / (self.args.hidden_size / self.args.dim_model_base))
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
