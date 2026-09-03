import dashscope
import sys

def call_llm(prompt):
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,
        prompt=prompt,
        result_format='message'
    )
    return response

if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]) or "你好"
    dashscope.api_key = "sk-ws-H.PMHEIHP.VITO.MEUCIQDf1IHo7RQW7UjW_0n8IuicYHIs6CNF3t-Si9hBFV2p5gIgDvLwzcQMgJ4_3BNRnseFvk_DHzEmeXkSIN5EcCqgBTA"
    result = call_llm(user_input)
    if result.status_code == 200:
        print(result.output.choices[0].message.content)
    else:
        print(f"请求失败: {result.code} - {result.message}")
