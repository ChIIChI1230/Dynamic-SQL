import os
import re
import sys
import copy
import json
import argparse
import concurrent.futures
from tqdm import tqdm
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM_CODER
from utils.simplified_schema import simplified, explanation_collection, simplified_ddl1
from utils import extract_tables_and_columns, get_all_schema

instruction1 = """
你是一个 SQL 查询生成助手，负责根据用户问题、数据库结构信息、与问题相关的领域知识、字段解释，生成准确、高效、可执行的 SQL 查询。

### 任务目标：
1. 理解用户的查询需求：  
    - 抽取查询目标（SELECT 列、聚合、分组等）、过滤条件（WHERE）、排序逻辑（ORDER BY）、连接关系（JOIN）等关键信息。  
2. 分析数据库结构信息：  
    - 根据表名、字段名、数据类型和外键信息，确定表之间的连接方式。  
3. 利用字段解释：  
    - 结合字段解释理解问题和数据库结构，确保条件表达更贴近用户语义。   
4. 生成准确、高效、可执行的 SQL 查询。

### 输入：
- 用户问题：自然语言形式的查询或问题；
- 数据库结构：包括表名、字段、表间关系（如外键等）；
- 字段解释：包括枚举值及其描述，用于理解数据列的含义和限制；
- 与问题相关的领域知识：用于优化用户问题和理解数据库结构。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请以 JSON 格式返回结果，格式如下：{{"sql": "完整的 SQL 语句"}}

### 操作步骤：
1. 从用户问题中提取 SELECT、FROM、JOIN、WHERE、GROUP BY、ORDER BY 等部分；
2. 根据外键信息补全 JOIN 子句；
3. 生成最终 SQL 并按要求的 JSON 格式返回。

### 注意事项：
- 始终使用反引号包裹表名和列名；
- 对日期、字符串等常量使用单引号；
- 如果重写后问题中已经足够完整，不要做额外假设。
"""

db_schema_copy = copy.deepcopy(get_all_schema())

with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
    column_meaning = json.load(f)
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
        print(f"Error extracting JSON with regex: {e}")

    # 2. 尝试最简单的 { … } 截取
    try:
        start = message.find("{")
        end   = message.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = message[start:end + 1]
            return _cleanup(candidate)
        else:
            print("未找到有效 JSON 区间，直接返回原始消息")
            return message
    except Exception as e:
        print(f"Error extracting JSON using find: {e}")
        return message

def generation_sql(schema, question, evidence, foreign_key, explanation):
    try:
        # table_info = (f'### Sqlite SQL tables, with their properties:\n{simplified_schema}\n'
        #               f'### Foreign key information of Sqlite SQL tables, used for table joins:\n{foreign_key}\n'
        #               f'### The meaning of every column:\n#\n {explanation}\n'
        #               f'#\n')
        table_info = (
            f"Sqlite SQL 表及其属性：\n{schema}\n"
            f"Sqlite SQL 表的外键信息，用于表连接：\n{foreign_key}\n"
            f"每一列的含义：\n{explanation}\n"
        )
        context = (
            f'用户问题: "{question}"\n'
            f'数据库结构：\n{table_info}\n'
        )
        if evidence:
            context += (f'与问题相关的领域知识：{evidence}\n')
        
        # print(context)
        
        llm = QWEN_LLM_CODER()
        # llm = SILICONFLOW()
        message = llm(instruction1, context)
        # print(message)
        message = message.replace("\\n", " ")
        message = message.replace("\\'", " ")
        # print(message)
        message = extract_json(message)
        
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            print("Warning: Failed to decode JSON from LLM response. Returning empty sql.")
            return [], []
        
        # print("message:" + message)

        sql = message.get('sql', '')
        return sql
    except Exception as e:
        print(f"Error in generation_sql: {e}")
        return [], []

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

def process_item(item):
    try:
        ### 收集解释
        explanation = ""
        for key, meaning in column_meaning.items():
            db_name = key.split("|")[0]
            table_name = key.split("|")[1]
            column_name = key.split("|")[2]
            if db_name == item['db']:
                explanation += f"### {table_name}.{column_name}: {meaning}\n"

        # 通过 LLM 抽取表和列信息
        sql = ""
        while len(sql) == 0:
            sql = generation_sql(item['simplified_ddl'], item['question'], item['evidence'], item['foreign_key'], explanation)
    
        entity = {
            "question_id": item.get('question_id', 'N/A'),
            "db": item.get('db',),
            "question": item.get('question', ''),
            "sql": sql,
            # "evidence": item.get('evidence', ''),
            # "simplified_ddl": item.get('simplified_ddl', ''),
            # "explanation": explanation,
            "difficulty": item.get('difficulty')
        }
        return entity
    except KeyError as e:
        print(f"Warning: Missing key {e} in item: {item}")
    except Exception as e:
        print(f"Error processing item: {e}")
    return None

def main(input_file, output_file, start_index, max_workers = 8):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                items = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error: Failed to load JSON from {input_file}. {e}")
                return
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error opening file {input_file}: {e}")
        return

    items_to_process = items[start_index:]
    
    # 使用 ThreadPoolExecutor 实现多线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
        open(output_file, 'w', encoding='utf-8') as f:
        # 提交所有任务
        futures = {executor.submit(process_item, item): item for item in items_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing schema link from LLM"):
            result = future.result()
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    print(f"Successfully saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_index", type=int, default = 0)
    parser.add_argument("--input_file", type=str, default="src/dataset/ppl_dev.json")
    parser.add_argument("--input_file1", type=str, default="src/dataset/ppl_dev_null.json")
    parser.add_argument("--output_file", type=str, default="src/dataset/qwen/coder-32b/baseline_sql.jsonl")
    parser.add_argument("--output_file1", type=str, default="src/dataset/qwen/coder-32b/baseline_sql_null.jsonl")
    parser.add_argument("--max_workers", type=int, default = 8, help="Number of worker threads")
    args = parser.parse_args()
    # extract_error_json(args.input_file, args.output_file, args.input_file1)
    main(args.input_file1, args.output_file1, args.start_index, args.max_workers)