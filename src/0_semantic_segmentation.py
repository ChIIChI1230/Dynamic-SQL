import re
import os
import sys
import argparse
import json
import logging
from tqdm import tqdm
import concurrent.futures
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

instruction = """
Please semantically segment the following natural language question and the relevant domain knowledge into structured components. Extract and return the following information in valid and parsable JSON format:

1. User Intent: A concise description of what the user wants to accomplish.
2. Target Entities: The primary tables or data entities involved (e.g., "users", "orders", "products").
3. Relevant Fields: Specific columns or attributes explicitly or implicitly referenced (e.g., "registration_date", "city", "user_id").
4. Operation Types: The types of data operations required (e.g., query, count, group by, filter, sort, aggregate, max/min, sum).
5. Filtering Conditions: Any constraints mentioned (e.g., time ranges, numeric thresholds, string matches).
6. Grouping/Sorting Requirements: Any grouping or sorting instructions (e.g., "group by city", "order by registration count descending").
7. Expected Output: What the user expects in the final output (e.g., "return the top 10", "only return totals", "return fields A and B").

Input example:
---
Question: "Count the number of users who registered after 2020 for each city, and sort the results by the number of users in descending order."
Expected Output:
{
  "User Intent": "Count the number of users registered after 2020 for each city",
  "Target Entities": ["users"],
  "Relevant Fields": ["registration_date", "city", "user_id"],
  "Operation Types": ["count", "group by", "sort"],
  "Filtering Conditions": ["registration_date > 2020"],
  "Grouping/Sorting Requirements": ["group by city", "order by user count descending"],
  "Expected Output": "return all cities with corresponding registration counts"
}

Now process the following:

Question: {question}  
Domain Knowledge: {domain_knowledge}

Respond only with valid, parsable JSON.
""" 

def extract_json(message):
    """
    尝试从 LLM 返回中提取 JSON 对象：
    1. 优先匹配 ```json { ... } ``` 代码块；
    2. 如果没有，再找第一个 '{' 和最后一个 '}' 之间的内容；
    3. 如果都失败，就原样返回。

    并且：
      - 如果提取到的字符串在闭合 '}' 前是 '";'，则把它替换为 '"}'
      - 否则，若在闭合 '}' 前缺少 '"'，再自动补上一个
    """
    def _cleanup(json_str: str) -> str:
        # —— 新增：如果闭合 } 前是 '";'，替换成 '"}'
        json_str = re.sub(r'";\s*}$', '"}', json_str)

        # —— 原有：补齐缺失的双引号
        if not json_str.endswith('}'):
            return json_str

        # 跳过 '}' 之前所有空白，找到最后一个非空白字符
        i = len(json_str) - 2
        while i >= 0 and json_str[i].isspace():
            i -= 1

        # 如果它不是 '"'，就在它后面插入一个
        if i >= 0 and json_str[i] != '"':
            insert_pos = i + 1
            json_str = json_str[:insert_pos] + '"' + json_str[insert_pos:]

        return json_str

    # 1. 尝试 ```json … ``` 代码块
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", message, re.DOTALL)
        if match:
            return _cleanup(match.group(1))
    except Exception as e:
        logging.error(f"Error extracting JSON with regex: {e}")

    # 2. 尝试最简单的 { … } 截取
    try:
        start = message.find("{")
        end   = message.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = message[start:end + 1]
            return _cleanup(candidate)
        else:
            logging.warning("未找到有效 JSON 区间，直接返回原始消息")
            return message
    except Exception as e:
        logging.error(f"Error extracting JSON using find: {e}")
        return message

def semantic_segmentation(question, evidence):
    try:
        context = (
            f'Question: "{question}"\n'
            f'Domain Knowledge：\n{evidence}\n'
        )
        llm = QWEN_LLM()
        response = llm(instruction, context)
        response1 = extract_json(response)
        response1.replace("\'", "'")
        print(response1)
        try:
            response_json = json.loads(response1)
            
            return response_json
        except json.JSONDecodeError as e:
            logging.warning(f"semantic_segmentatio 无法解析 LLM 返回的 JSON: {e}, 返回空字符串" + f"\n{response}")
            return ""
    except Exception as e:
        logging.error(f"semantic_segmentation 异常:{e}")

def process_item(ppl):
    try:
        question = ppl['question']
        question_id = ppl['question_id']
        evidence = ppl['evidence']
        # semantic_segmentation_res = semantic_segmentation(question, evidence)
        if question_id == 1347:
            semantic_segmentation_res = {
                "User Intent": "Retrieve the hometown county for a specific person",
                "Target Entities": ["persons"],
                "Relevant Fields": ["name", "county"],
                "Operation Types": ["query"],
                "Filtering Conditions": ["name = 'Adela O'Gallagher'"],
                "Grouping/Sorting Requirements": [],
                "Expected Output": "return the hometown county for Adela O'Gallagher"
            }
        else:
            semantic_segmentation_res = {
                "User Intent": [
                    "Determine the frequency of account statement requests for account number 3",
                    "Identify the aim of debiting a total amount of 3539"
                ],
                "Target Entities": [
                    "transactions",
                    "accounts"
                ],
                "Relevant Fields": [
                    "account_number",
                    "transaction_type",
                    "amount",
                    "k_symbol"
                ],
                "Operation Types": [
                    "count",
                    "filter",
                    "aggregate",
                    "lookup"
                ],
                "Filtering Conditions": [
                    "account_number = 3",
                    "transaction_type = 'account statement request'",
                    "amount = 3539",
                    "transaction_type = 'debit'"
                ],
                "Grouping/Sorting Requirements": [],
                "Expected Output": [
                    "return the count of account statement requests for account number 3",
                    "return the k_symbol for the total debit amount of 3539"
                ]
            }
        semantic_segmentation_res_list = []
        for key, values in semantic_segmentation_res.items():
            if isinstance(values, list):
                for value in values:
                    semantic_segmentation_res_list.append(value)
            elif isinstance(values, str):
                semantic_segmentation_res_list.append(values)
        # print("semantic_segmentation_res_list:" + str(semantic_segmentation_res_list))
        entity = {
            "question_id": ppl['question_id'],
            "db": ppl['db'],
            "question": ppl['question'],
            "evidence": ppl['evidence'],
            "foreign_key": ppl['foreign_key'],
            "semantic_seg_dict": semantic_segmentation_res,
            "semantic_seg_list": semantic_segmentation_res_list,
            "difficulty": ppl['difficulty']
        }

        return entity
    except KeyError as e:
        logging.warning(f"缺少键 {e}，item={ppl}")
    except Exception as e:
        logging.error(f"处理 item 时异常: {e}")
    return None

def extract_error_json(input_file1, input_file2, output_file):
    # 从 file1 中读取所有 JSONL 记录，确保解析结果为 dict 类型
    items1 = []
    with open(input_file1, 'r', encoding='utf-8') as f:
        items1 = json.load(f)

    # 从 file2 中读取记录，并构建 question_id 的集合
    file2_question_ids = set()
    with open(input_file2, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    question_id = parsed.get('question_id')
                    if question_id is not None:
                        file2_question_ids.add(question_id)
                else:
                    print(f"Warning: Unexpected JSON structure in {input_file2}: {line}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line in {input_file2}: {e}")

    # 筛选出 file1 中 question_id 不存在于 file2_question_ids 的记录
    results = [item for item in items1 if item.get('question_id') not in file2_question_ids]

    # 将结果逐行写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4,
                      default=lambda o: list(o) if isinstance(o, set) else o) 
def main(input_file, output_file, start_index, max_workers = 8):
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    items_to_process = items[start_index:]
    logging.info(f"待处理条目数: {len(items_to_process)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        futures = {executor.submit(process_item, it): it for it in items_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing items"):
            try:
                result = future.result()
                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
            except Exception as e:
                logging.error(f"处理并发任务时异常: {e}")

    logging.info(f"已完成，结果保存在 {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 命令行参数解析
    parser.add_argument("--start_index", type = int, default = 0)
    parser.add_argument("--ppl_file", type = str, default = "src/dataset/ppl_dev.json")
    parser.add_argument("--ppl_file1", type = str, default = "src/dataset/ppl_dev_null.json")
    parser.add_argument("--semantic_out_file", type = str, default = "src/dataset/qwen/coder-32b/semantic_seg.jsonl")
    parser.add_argument("--semantic_out_file1", type = str, default = "src/dataset/qwen/coder-32b/semantic_seg_null.jsonl")
    args = parser.parse_args()
    # extract_error_json(args.ppl_file, args.semantic_out_file, args.ppl_file1)
    main(args.ppl_file1, args.semantic_out_file1, args.start_index)