"""
采样参数方面：temperature 越低，输出越稳定、重复性越强；temperature 较高时，表达更丰富，但更容易跑题。
top_p 限制候选词范围，较低时更保守，较高时更开放。分类、抽取等任务适合低温或贪心解码；创意写作、头脑风暴适合适当提高温度。
提示策略方面：Zero-shot 简单快速，但格式和边界可能不稳定；Few-shot 能显著提高格式一致性和标签选择稳定性；
Chain-of-Thought 对复杂判断更有帮助，但会增加 token 成本，也可能输出冗余分析。客服意图分类这种任务通常 Few-shot 效果最好。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)
device = next(model.parameters()).device


def ask(prompt, generation_config, enable_thinking=False):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        **generation_config
    )
    new_tokens = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


sample_prompt = "请用三句话介绍本地部署开源大语言模型的优点和缺点。"

sampling_tests = {
    "greedy": {"do_sample": False},
    "low_temperature": {"do_sample": True, "temperature": 0.2, "top_p": 0.8},
    "balanced": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    "creative": {"do_sample": True, "temperature": 1.1, "top_p": 0.95},
}

print("=== 采样参数对比 ===")
for name, config in sampling_tests.items():
    print(f"\n--- {name} ---")
    print(ask(sample_prompt, config))


task_text = "我昨天买的耳机一直没有发货，客服也没人回复，我现在想退款。"

zero_shot = f"""
请判断下面用户消息的客服意图，只输出JSON。
可选意图：物流查询、退款申请、商品咨询、投诉、其他。
用户消息：{task_text}
"""

few_shot = f"""
请判断用户消息的客服意图，只输出JSON。

示例1：
用户消息：我的快递到哪里了？
输出：{{"意图":"物流查询"}}

示例2：
用户消息：这个手机支持无线充电吗？
输出：{{"意图":"商品咨询"}}

现在判断：
用户消息：{task_text}
"""

cot = f"""
请先简要分析用户的核心诉求，再给出最终JSON。
可选意图：物流查询、退款申请、商品咨询、投诉、其他。
用户消息：{task_text}
"""

print("\n=== 提示策略对比 ===")
for name, prompt in {
    "Zero-shot": zero_shot,
    "Few-shot": few_shot,
    "Chain-of-Thought": cot,
}.items():
    print(f"\n--- {name} ---")
    print(ask(prompt, {"do_sample": False}, enable_thinking=False))