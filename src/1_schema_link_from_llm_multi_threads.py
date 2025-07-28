import os
import re
import sys
import copy
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from instruction import SQL_GENERATION_INSTRUCTION1
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM_CODER, DP_LLM, GPT_LLM
from utils.simplified_schema import simplified, explanation_collection, simplified_ddl1
from utils import extract_tables_and_columns, get_all_schema

with open('src/dataset/ppl_dev.json', "r",  encoding="utf-8") as f:
    ppl_dev = json.load(f)

ppl_dev_array = []
for item in ppl_dev:
    ppl_dev_array.append(item['simplified_ddl'])

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

reduce = 0

db_schema_copy = copy.deepcopy(get_all_schema())
def prefect_foreign_key(tables, columns, foreign_key):
    new_tables = list(tables)
    new_columns = list(columns)
    for line in foreign_key.split("\n"):
        try:
            # 去掉前缀 #
            line = line.lstrip("#").strip()

            # 拆分 references 左右两侧
            left, right = line.split(" references ")

            # 解析左侧部分
            table1 = left.split("(")[0].strip()
            column1 = left.split("(")[1].split(")")[0].strip()

            # 解析右侧部分
            table2 = right.split("(")[0].strip()
            column2 = right.split("(")[1].split(")")[0].strip()

            # 只在两个表都合法的情况下，才处理和添加
            if table1.lower() in new_tables and table2.lower() in new_tables:
                if f"{table1}.{column1}" not in new_columns:
                    new_columns.append(f"{table1}.{column1}")
                if f"{table2}.{column2}" not in new_columns:
                    new_columns.append(f"{table2}.{column2}")
        except:
            continue
    
    return new_tables, new_columns

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
            f"### Sqlite SQL tables, with their properties:\n{schema}\n"
            f"### Foreign key information of Sqlite SQL tables, used for table joins:\n{foreign_key}\n"
            f"explanation：\n{explanation}\n"
        )
        context = (
            f'\n### Question: "{question}"\n'
            f'{table_info}\n'
        )
        if evidence:
            context += (f'### definition: {evidence}\n')
        
        prompt = SQL_GENERATION_INSTRUCTION1 + context
        return prompt
        # llm = QWEN_LLM_CODER()
        # llm = SILICONFLOW()
        # llm = DP_LLM()
        # llm = GPT_LLM()
        # message = llm(SQL_GENERATION_INSTRUCTION1, context)
        # # message = message.replace("\\n", " ")
        # # message = message.replace("\\'", " ")
        # print(message)
        # message = extract_json(message)
        
        # try:
        #     message = json.loads(message)
        # except json.JSONDecodeError:
        #     print("Warning: Failed to decode JSON from LLM response. Returning empty sql.")
        #     return [], []
        
        # # print("message:" + message)

        # sql = message.get('sql', '')
        # # sql = message
        # return sql
    except Exception as e:
        print(f"Error in generation_sql: {e}")
        return [], []



def process_item(item):
    global reduce
    try:
        # 根据匹配到的表名简化模式
        table_list = item['tables']
        column_list = item['columns']

        # simplified_ddl 函数负责生成简化的 schema 和 foreign_key
        simplified_schema, foreign_key, explanation, column_list = simplified_ddl1(item['db'], table_list, column_list, item['foreign_key'])
        
        # 通过 LLM 抽取表和列信息
        # sql = "WITH TallestPlayers AS (    SELECT player_api_id, finishing    FROM Player p    JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id    WHERE height = (SELECT MAX(height) FROM Player)),ShortestPlayers AS (    SELECT player_api_id, finishing    FROM Player p    JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id    WHERE height = (SELECT MIN(height) FROM Player)), AverageFinishingRates AS (    SELECT 'Tallest' AS group_type, AVG(finishing) AS avg_finishing    FROM TallestPlayers    UNION ALL    SELECT 'Shortest' AS group_type, AVG(finishing) AS avg_finishing    FROM ShortestPlayers), MaxFinishingRateGroup AS (    SELECT group_type    FROM AverageFinishingRates    ORDER BY avg_finishing DESC    LIMIT 1)SELECT p.player_name FROM Player p JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id JOIN (    SELECT player_api_id    FROM TallestPlayers    WHERE (SELECT group_type FROM MaxFinishingRateGroup) = 'Tallest'    UNION ALL    SELECT player_api_id    FROM ShortestPlayers    WHERE (SELECT group_type FROM MaxFinishingRateGroup) = 'Shortest') AS MaxFinishingRatePlayers ON p.player_api_id = MaxFinishingRatePlayers.player_api_id ORDER BY pa.finishing DESCLIMIT 1"
        sql = ""
        while len(sql) == 0:
            sql = generation_sql(simplified_schema, item['question'], item['evidence'], foreign_key, explanation)
        print(sql)
        ans = extract_tables_and_columns(sql)

        columns0 = []
        for table in ans['table']:
            for column in ans['column']:
                schema = f"{table}.{column}"
                list_db = [item.lower() for item in db_schema_copy[item['db']]]
                if schema.lower() in list_db:
                    columns0.append(schema)
        
        # 合并 LLM 和原始数据中的结果
        tables = list(set(ans['table']) | set(item['tables']))
        columns = list(set(columns0) | set(column_list))
        tables, columns = prefect_foreign_key(tables, columns, item['foreign_key'])
        
        item['tables'] = tables
        item['columns'] = columns

        # simplified 函数用于构建最终数据库结构
        final_schema, final_foreign_key = simplified(item)
        # ddl1 = ppl_dev_array[item.get('question_id')]
        # prompt0 =  generation_sql(ddl1, item['question'], item['evidence'], foreign_key, explanation)
        # prompt1 =  generation_sql(simplified_schema, item['question'], item['evidence'], foreign_key, explanation)
        # len0 = len(prompt0)
        # len1 = len(prompt1)

        # reduce += 1 - (len1/(2*len0))

        entity = {
            "question_id": item.get('question_id', 'N/A'),
            # "prompt0": len(prompt0),
            # "prompt1": len(prompt1)
            "db": item.get('db', 'unknown'),
            "question": item.get('question', ''),
            "sql_1": sql,
            "evidence": item.get('evidence', ''),
            "llm_tables": list(ans['table']),
            "llm_columns": columns0,
            "tables": tables,
            "columns": columns,
            "final_schema": final_schema,
            "final_foreign_key": final_foreign_key,
            "difficulty": item.get('difficulty')
        }

        return entity
    except KeyError as e:
        print(f"Warning: Missing key {e} in item: {item}")
    except Exception as e:
        print(f"Error processing item: {e}")
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
    print(reduce / 1534)
    print(f"Successfully saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default = 0)
    parser.add_argument("--input_file", type=str, default="src/dataset/qwen/coder-32b/sl_out_milvus_new2.json")
    parser.add_argument("--input_file1", type=str, default="src/dataset/qwen/coder-32b/sl_out_milvus_new2_null.json")
    parser.add_argument("--output_file", type=str, default="src/dataset/qwen/coder-32b/1_sl_final_coder.jsonl")
    parser.add_argument("--prompt_output_file", type=str, default="src/dataset/qwen/coder-32b/prompt_1.jsonl")
    parser.add_argument("--output_file1", type=str, default="src/dataset/qwen/coder-32b/1_sl_final_coder1_null.jsonl")
    parser.add_argument("--max_workers", type=int, default = 8, help="Number of worker threads")
    args = parser.parse_args()
    # extract_error_json(args.input_file, args.output_file, args.input_file1)
    main(args.input_file, args.prompt_output_file, args.start_index, args.max_workers)