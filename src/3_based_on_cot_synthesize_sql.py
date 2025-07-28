import os
import re
import sys
import json
import argparse
import logging
from tqdm import tqdm
import concurrent.futures
from instruction import COT_SYNTHESIZE_SQL_INSTRUCTION

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM_CODER
from utils.util import execute_sql

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

instruction = """
你是一位精通 SQL 语义理解与结构优化的专家，擅长根据用户查询意图，分析、对比并融合多条 SQL 查询，生成结构合理、语义精准、执行正确的最终查询语句。

### 核心任务：
1. 比较两条 SQL 查询的结构、条件逻辑与执行结果，分析它们是否准确表达了用户意图；
2. 如果发现以下问题，需进行修正：
    - 字段选择错误或遗漏；
    - 过滤条件存在偏差或逻辑漏洞；
    - 查询结果出现数据缺失、冗余或顺序错误；
3. 总结每条 SQL 查询的正确、错误和冗余之处；
4. 尽可能复用有效部分，融合两条 SQL 的正确逻辑，修正偏差，消除重复或无效信息；
5. 保证最终输出的 SQL 语句结构清晰，语义准确，避免冗余字段或无用逻辑，并确保其可被数据库正确执行。

### 输入：
- 用户问题：自然语言描述的查询意图；
- SQL 查询及其执行结果列表：用于判断其与查询意图的一致性；
- 数据库结构信息：表名、字段名、字段类型、 表间关系等；
- 字段解释与枚举值：帮助理解字段含义与可选值；
- 相关列样本数据：每列的示例值、空值、格式异常等；
- 与问题相关的领域知识（如有）：对业务背景的补充说明。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请仅以 JSON 格式返回最终融合优化后的 SQL 查询，格式如下：{"sql": "最终合成的 SQL 查询语句"}

### 操作步骤：
1. 分析 SQL_1 是否存在字段选择错误、条件逻辑偏差或结构不合理的问题；
2. 分析 SQL_2 是否存在字段选择错误、条件逻辑偏差或结构不合理的问题；
3. 比较两条 SQL 的执行结果与用户问题的语义匹配度，识别各自的正确性和覆盖范围；
4. 在复用有效结构的基础上，融合两条 SQL 查询，剔除冗余或冲突内容；
5. 输出结构清晰、语义准确、结果可用的 SQL 查询语句。

### 注意事项
- 避免完全重写 SQL，优先保留并复用两条 SQL 中的有效结构和正确逻辑；
- 如果其中一条 SQL 查询已完全满足用户意图，可直接返回该条 SQL 作为最终结果；
- 输出必须遵循 JSON 格式，只返回 SQL 查询语句，禁止附加任何解释或推理过程。
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

def execute_single_sql(db_name, sql):
    try:
        row_count, column_count, result_preview, exec_time = execute_sql(sql, db_name)
    except Exception as e:
        logging.error(f"SQL 执行异常，数据库: {db_name}, SQL: {sql}. 错误: {e}")
        return {
            "sql": sql,
            "isvalid": False,
            "exec_time": 0,
            "error": f"exception: {e}"
        }
    if re.match(r"^(error:|timeouterror:)", result_preview, re.IGNORECASE):
        logging.info(f"SQL 执行失败: {sql}，错误信息: {result_preview}")
        return {
            "sql": sql,
            "isvalid": False,
            "exec_time": exec_time,
            "error": result_preview
        }
    else:
        logging.info(f"SQL 执行成功: {sql}")
        return {
            "sql": sql,
            "isvalid": True,
            "row_count": row_count,
            "column_count": column_count,
            "result_preview": result_preview,
            "exec_time": exec_time,
            "result": result_preview
        }

def cot_synthesize_sql(question, schema, foreign_key, evidence, explanation, data, sql, result):
    table_info = ('### Sqlite SQL tables, with their properties:\n' + schema +
                  '\n### Here are some data information about database references.\n' + data +
                  '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key +
                  '\n### The meaning of every column:\n#\n' + explanation +
                  '\n#\n')

    # sql_result = ""
    # for i in range(len(sql)):
    #     sql_result += f"SQL_{i + 1}: {sql[i]} 执行结果: {result[i]['result'] if result[i]['isvalid'] else result[i]['error']}\n"
    # context = (
    #     f'用户问题: "{question}"\n'
    #     f'当前 SQL 查询及其执行结果：\n{sql_result}\n'                                                                                    
    #     f'数据库结构信息：\n{table_info}\n'
    #     f'字段解释与枚举值：\n{explanation}\n'                                                    
    #     f'相关列的样本数据：\n{data}\n'
    # )                                                                                                                  

    sql_result = ""
    for i in range(len(sql)):
        sql_result += f"SQL_{i + 1}: {sql[i]} Execution result: {result[i]['result'] if result[i]['isvalid'] else result[i]['error']}\n                                                                                                                                             "

    # context = "\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info + '\n\n' + '### definition: ' + evidence + "\n### Question: " + question

    # context += "\n### List of current SQL queries and their execution results:\n" + sql_result

    context = (
        "\n### Answer the question by sqlite SQL query only and with no explanation. "
        "You must minimize SQL execution time while ensuring correctness.\n"
        "### In the final SQL output, all table names and column names must be enclosed in backticks, like `table_name`.`column_name`.\n"
        + table_info +
        '\n### definition:\n' + evidence + 
        '\n### Question:\n' + question +
        '\n### List of current SQL queries and their execution results:\n' + sql_result
    )
    
    try:
        llm = QWEN_LLM_CODER()
        response = llm(COT_SYNTHESIZE_SQL_INSTRUCTION, context)
        logging.info("语义对齐 LLM 返回: " + response)
        response1 = extract_json(response)
        try:
            response_json = json.loads(response1)
            return response_json.get("sql", "")
        except json.JSONDecodeError as e:
            logging.warning(f"无法解析 LLM 返回的 JSON: {e}, 返回空字符串" + f"\n{response1}")
            return ""
    except Exception as e:
        logging.error(f"semantic_alignment 调用 LLM 异常: {e}")
        return ""

def process_item(item):
    try:
        db = item.get("db")
        question = item.get("question")
        evidence = item.get("evidence")
        schema = item.get("schema")
        foreign_key = item.get("foreign_key")
        explanation = item.get("explanation")
        data = item.get("data")
        sql_1 = item.get("sql_1")
        sql_2 = item.get("sql_2")

        result1 = execute_single_sql(db, sql_1)
        result2 = execute_single_sql(db, sql_2)

        sql = [sql_1, sql_2]
        result = [result1, result2]

        sql_3 = cot_synthesize_sql(question, schema, foreign_key, evidence, explanation, data, sql, result)
        
        entity = {
            "question_id": item.get("question_id"),
            "db": db,
            "question": question,
            "sql_1": sql_1,
            "sql_2": sql_2,
            "sql_3": sql_3,
            "evidence": evidence,
            "schema": schema,
            "foreign_key": foreign_key,
            "explanation": explanation,
            "data": data,
            "difficulty": item.get("difficulty")
        }
        return entity
    except KeyError as e:
        logging.warning(f"缺少键 {e}，item={item}")
    except Exception as e:
        logging.error(f"处理 item 时异常: {e}")
    return None

def main(input_file, output_file, start_index, max_workers=8):
    items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"解码 JSON 行时错误: {e}")
    except Exception as e:
        logging.error(f"读取文件 {input_file} 异常: {e}")
        return

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
    parser.add_argument("--start_index", type = int, default = 0)
    parser.add_argument("--input_file", type = str, default = "src/dataset/qwen/coder-32b/en/2_sql_generation1.jsonl")
    parser.add_argument("--output_file", type = str, default = "src/dataset/qwen/coder-32b/en/3_cot_synthesize_sql1.jsonl")
    parser.add_argument("--max_workers", type = int, default = 8, help = "线程数")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.start_index, args.max_workers)