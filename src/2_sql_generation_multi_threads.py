import os
import sys
import re
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from instruction import SQL_GENERATION_INSTRUCTION

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM_CODER

instruction = """
你是一个 SQL 查询生成助手，擅长将自然语言问题转化为高质量、可执行的 SQL 语句。请结合数据库结构信息、字段解释、样本数据等内容，完成以下任务。

### 任务目标：
根据提供的信息生成结构正确、语义准确、可执行的 SQL 查询，满足用户的问题意图。SQL 应涵盖：
- 查询字段（SELECT）
- 表和连接（FROM / JOIN）
- 过滤条件（WHERE）
- 分组与聚合（GROUP BY / HAVING）
- 排序与限制（ORDER BY / LIMIT）

### 输入：
- 用户问题：自然语言描述的查询意图；
- 数据库结构信息：表名、字段名、字段类型、 表间关系等；
- 字段解释与枚举值：字段含义、枚举值描述等；
- 相关列样本数据：每列的示例值、空值、格式异常等；
- 与问题相关的领域知识（如有）：对业务背景的补充说明。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请以 JSON 格式返回结果，格式如下：{{"sql": "完整的 SQL 语句"}}

### 示例：
- 用户问题：{question}
- sql语句：{sql}

### 操作步骤：
1. 分析用户问题，明确需求并拆解对应的 SQL 结构。
2. 依据数据库外键关系，合理补全 JOIN 子句，确保表间连接正确。
3. 根据字段解释与枚举值，将自然语言表述的状态、分类等信息映射为准确的枚举值或代码，填入 WHERE 条件。
4. 参考样本数据，若字段存在空值、格式异常等情况，应自动添加必要的过滤条件，避免查询报错或结果不准确。
5. 生成完整 SQL 查询语句，并按上述 JSON 格式返回。

### 注意事项：
- 始终使用反引号包裹表名和列名；
- 对日期、字符串等常量使用单引号。
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

def generation_sql(question, schema, foreign_key, evidence, explanation, example, data):
    # table_info = (
    #     f"Sqlite SQL 表及其属性：\n{schema}\n"
    #     f"Sqlite SQL 表的外键信息，用于表连接：\n{foreign_key}\n"
    #     f"每一列的含义：\n{explanation}\n"
    # )
    # context1 = (
    #     f'重写后的用户问题: "{question}"\n'
    #     f'数据库结构信息：\n{table_info}\n'
    # )
    # context2 = (
    #     f'字段解释与枚举值：\n{explanation}\n'
    #     f'相关列的样本数据：\n{data}\n'
    # )

    table_info = ('### Sqlite SQL tables, with their properties:\n' + schema +
                  '\n### Here are some data information about database references.\n' + data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key +
                  '\n### The meaning of every column:\n#\n' + explanation +
                  '\n#\n')

    context = "### example:\n" + "Question:"+ example.get('question') + "sql:" + example.get('sql') + "\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question

    try:
        # example = json.loads(example)
        # instruction1 = instruction.format(question = example['question'], sql = example['sql'])
        llm = QWEN_LLM_CODER()
        print(context)
        response = llm(SQL_GENERATION_INSTRUCTION, context)
        response1 = extract_json(response)
        # response1 = response1.replace("Women\'s Soccer", "Women''s Soccer")
        # print(response1)
        # return response1
        try:
            response_json = json.loads(response1)
            return response_json.get("sql", "")
            
        except json.JSONDecodeError:
            print("Warning: 无法解析 LLM 返回的 JSON，返回空字符串。")
            return ""
    except Exception as e:
        print(f"Error in generation_sql: {e}")
        return ""

def extract_error_json(input_file1, input_file2, output_file):
    # 从 file1 中读取所有 JSONL 记录，确保解析结果为 dict 类型
    items1 = []
    with open(input_file1, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items1.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

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
                    sql_2 = parsed.get('sql_2')
                    if question_id is not None and sql_2 != "":
                        file2_question_ids.add(question_id)
                else:
                    print(f"Warning: Unexpected JSON structure in {input_file2}: {line}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line in {input_file2}: {e}")

    # 筛选出 file1 中 question_id 不存在于 file2_question_ids 的记录
    results = [item for item in items1 if item.get('question_id') not in file2_question_ids]

    # 将结果逐行写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            # 移除 indent 参数，添加换行符
            json_line = json.dumps(
                item,
                ensure_ascii=False,
                default=lambda o: list(o) if isinstance(o, set) else o
            )
            f.write(json_line + '\n')  # 显式添加换行符

def process_item(item):
    try:
        question = item.get("question")
        evidence = item.get("evidence")
        db = item.get("db")
        columns = item.get("columns")
        schema = item.get("schema")
        foreign_key = item.get("foreign_key")
        explanation = item.get("explanation")
        example = item.get("explame")
        data = item.get("data")
        
        attempts = 0
        sql = ""
        while sql == "" and attempts < 3:
            sql = generation_sql(
                question,
                schema,
                foreign_key,
                evidence,
                explanation,
                example,
                data
            )
            attempts += 1
        
        return {
            "question_id": item.get("question_id"),
            "db": db,
            "question": question,
            "evidence": evidence,
            "columns": columns,
            "sql_1": item.get("sql_1"),
            "sql_2": sql,
            "schema": schema,
            "foreign_key": foreign_key,
            "explanation": explanation,
            "data": data,
            "example": example,
            "difficulty": item.get("difficulty")
        }
    except KeyError as e:
        print(f"Warning: 缺少键 {e}，item={item}")
    except Exception as e:
        print(f"Error processing item: {e}")
    return None

def main(input_file, output_file, start_index, max_workers = 8):
    # 按行读取 JSONL
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")

    items_to_process = items[start_index:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
         open(output_file, 'w', encoding='utf-8') as out_f:

        futures = {executor.submit(process_item, it): it for it in items_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing items"):
            result = future.result()
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"已完成，结果保存在 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 命令行参数解析
    parser.add_argument("--start_index", type = int, default = 1530)
    parser.add_argument("--input_file", type = str, default = "src/dataset/qwen/coder-32b/en/1_5_normalize_schema1.jsonl")
    parser.add_argument("--output_file", type = str, default = "src/dataset/qwen/coder-32b/en/2_sql_generation2.jsonl")
    parser.add_argument("--input_file1", type = str, default = "src/dataset/qwen/coder-32b/en/1_5_normalize_schema_null.jsonl")
    parser.add_argument("--output_file1", type = str, default = "src/dataset/qwen/coder-32b/en/2_sql_generation1_null.jsonl")
    parser.add_argument("--max_workers", type=int, default = 8, help = "线程数")
    args = parser.parse_args()

    # extract_error_json(args.input_file, args.output_file, args.input_file1)
    main(args.input_file, args.output_file, args.start_index, args.max_workers)