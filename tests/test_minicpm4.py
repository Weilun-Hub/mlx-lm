from mlx_lm import load, generate

MODEL_PATH = "/Users/macmini24g/Workspace/zhaoweilun/models"
MODEL_NAME = "MiniCPM4-8B"

if __name__ == "__main__":

    model, tokenizer = load(f"{MODEL_PATH}/{MODEL_NAME}")
    # prompt = "Tell me who you are"
    prompt = "Write an article about Artificial Intelligence" * 2000
    

    text = generate(model, tokenizer, prompt, max_tokens=128, verbose=True)

    print(text)
