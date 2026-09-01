import os
from openai import OpenAI


class DeepSeekClient:
    """
    一个用于调用 DeepSeek API 的客户端。
    DeepSeek 兼容 OpenAI SDK，所以仍然使用 openai 包。
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str = None,
        base_url: str = "https://api.deepseek.com"
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url
        )

    def generate(self, prompt: str, system_prompt: str = "你是一个有帮助的助手。") -> str:
        """调用 DeepSeek API 生成回应。"""
        print("正在调用 DeepSeek 大语言模型...")

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )

            answer = response.choices[0].message.content
            print("DeepSeek 响应成功。")
            return answer

        except Exception as e:
            print(f"调用 DeepSeek API 时发生错误: {e}")
            return "错误：调用 DeepSeek 语言模型服务时出错。"