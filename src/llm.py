import openai
import json
import requests
from config import QWEN
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

class QWEN_LLM:
    def __init__(self):
        self.client = openai.OpenAI(api_key = QWEN.api, base_url = QWEN.base_url)

    def __call__(self, instruction, prompt):
        num = 0
        max_retries = 3  # 限制最大重试次数
        response = None

        while num < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model = QWEN.model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=0
                )

                # 解析 JSON，确保返回的是有效 JSON
                content = response.choices[0].message.content
                return content  # 解析成功，返回内容

            except json.JSONDecodeError:
                print(f"Warning: Response is not a valid JSON. Retrying {num + 1}/{max_retries}...")
                num += 1
                

            except Exception as e:
                print(f"Error: {e}. Retrying {num + 1}/{max_retries}...")
                num += 1
                
        return None  # 失败后返回 None，防止后续代码崩溃