from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

if __name__ == "__main__":

    model_path = "/Users/macmini24g/Workspace/zhaoweilun/models/MiniCPM4-8B"

    model_files = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    layers_to_remove = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    weights = {}
    for file in model_files:
        with safe_open(f"{model_path}/{file}", framework="pt") as f:
            for key in f.keys():
                flag = False
                for layer in layers_to_remove:
                    if f"model.layers.{layer}" in key:
                        flag = True
                        break
                if flag:
                    continue
                weights[key] = f.get_tensor(key).to(torch.float16)

    print(weights.keys())
    save_file(weights, f"{model_path}/model.safetensors")