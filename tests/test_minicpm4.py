from mlx_lm import load, generate

MODEL_PATH = "/Users/macmini24g/Workspace/zhaoweilun/models"
MODEL_NAME = "MiniCPM4-8B"

def make_input(digits, before_len=2000, after_len=4000):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * before_len
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * after_len
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query

if __name__ == "__main__":

    # prompt = "Tell me who you are"
    # prompt = "Write an article about Artificial Intelligence. "  * 2000

    ans = 76384
    print("[DEBUG ZWL] gt answer:", ans)

    #before_len = 100 # 6066
    #after_len = 200

    #before_len = 170 # 8066
    #after_len = 270

    #before_len = 170 * 2 # 17666
    #after_len = 270 * 2

    #before_len = 155 * 4 # 32866
    #after_len = 255 * 4

    before_len = 100 * 8 # 48066
    after_len = 200 * 8

    prompt = make_input(ans, before_len, after_len)

    model, tokenizer = load(f"{MODEL_PATH}/{MODEL_NAME}")
    
    text = generate(model, tokenizer, prompt, max_tokens=128, verbose=True)

