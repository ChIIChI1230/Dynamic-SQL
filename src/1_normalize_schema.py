import os
import sys
import re
import json
import argparse
import sqlite3
import concurrent.futures
from tqdm import tqdm
from pymilvus import RRFRanker
from pymilvus import AnnSearchRequest
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from schema_link_from_milvus import get_vector

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from config import DEV

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

client = MilvusClient(
    uri = MILVUS_URI,
    token = MILVUS_TOKEN
)

# 全局初始化 embedding 函数
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name = 'BAAI/bge-m3',
    device = 'cuda:0',
    use_fp16 = True
)

with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
    column_meaning = json.load(f)

def normalize_column_name(db, columns):
    with open("data/dev_columns.json", "r", encoding="utf-8") as f:
        all_columns = json.load(f)
    columns_list = {col.lower(): col for col in all_columns[db]}
    columns_res = set()
    tables_res = set()
    for column in columns:
        if column.lower() in columns_list:
            columns_res.add(columns_list[column.lower()])
            tables_res.add(columns_list[column.lower()].split(".")[0])
    return list(tables_res), list(columns_res)

def build_schema(db, tables, columns):
    with open("data/foreign_key.json", "r", encoding="utf-8") as f:
        all_foreign = json.load(f)
    foreign = all_foreign[db]
    schema = ""
    foreign_key = ""
    
    temp = "#\n"
    for line in foreign.split("\n"):
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
            if table1.lower() in tables and table2.lower() in tables:
                if f"{table1}.{column1}" not in columns:
                    columns.append(f"{table1}.{column1}")
                if f"{table2}.{column2}" not in columns:
                    columns.append(f"{table2}.{column2}")
                temp += line + "\n"
        except:
            continue
    foreign_key = temp.strip() + "\n# "

    schema = "#\n# "
    for table in tables:
        schema += table + "("
        # 找出属于当前表的所有列，并提取列名（即点号后面的部分）
        cols = [col.split('.')[1] for col in columns if col.split('.')[0] == table]
        schema += ", ".join(cols)
        schema += ")\n# "
    schema = schema.strip()

    return schema, foreign_key, columns

def build_explanation(db, columns):
    explanation = ""
    for col in columns:
        parts = col.split(".")
        if len(parts) != 2:
            continue  # 或者记录错误信息
        table = parts[0].strip()
        column = parts[1].strip()
        meaning = column_meaning.get(f"{db}|{table}|{column}")
        if meaning is not None:
            explanation += f"### {table}.{column}: {meaning}\n"
    
    return explanation

def build_data(db, columns, question, evidence):
    file_path = f"milvus/sparse/count/{db}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        count_data = json.load(f)
    
    all_data = {}
    for col in columns:
        parts = col.split(".")
        if len(parts) != 2:
            continue  # 或者记录错误信息
        table_name = parts[0].strip()
        column_name = parts[1].strip()
        if len(count_data[table_name][column_name]["values"]) > 0:
            # 如果在向量数据库中未进行值存储，则从数据库中获取
            data = get_five_row_data(db, table_name, column_name)
        else:
            # 基于稀疏向量，从向量数据库中找到最相关的五列数据
            data = get_data_from_milvus_sparse(db, table_name, column_name, question, evidence)
            if len(data) == 0:
                data = get_five_row_data(db, table_name, column_name)
        
        all_data.setdefault(table_name, {})[column_name] = data

    
    results = ""
    for category, data_dict in all_data.items():
        category_str = f"# {category}("
        columns = []
        
        for column, values in data_dict.items():
            # 处理空值和异常数据
            if not isinstance(values, list):
                values = [values]
                
            # 安全提取元素值
            elements = []
            for v in values:
                try:
                    # 处理元组/列表和标量值
                    element = v[0] if isinstance(v, (tuple, list)) and len(v) > 0 else v
                except (IndexError, TypeError):
                    element = v  # 直接使用原始值
                
                # 转换None值为空字符串
                elements.append(str(element) if element is not None else "")
            
            # 构建列字符串
            column_str = f"{column}({','.join(elements)})"
            columns.append(column_str)
        
        # 合并类别内容
        category_str += ",".join(columns) + ");\n"
        results += category_str
    return results

def get_five_row_data(db, table, column):
    mydb = sqlite3.connect(DEV.dev_databases_path + '/' + db + f'/{db}.sqlite')
    cur = mydb.cursor()

    try:
        cur.execute(
            f"SELECT DISTINCT `{column}` "
            f"FROM `{table}` "
            f"ORDER BY RANDOM() "   # SQLite 用 RANDOM()；MySQL 请改为 RAND()
            f"LIMIT 5"
        )
        rows = cur.fetchall()
    except Exception as e:
        print(f"Warning: 随机抽样失败，降级到 LIMIT 查询 for {table}.{column}: {e}")
        try:
            cur.execute(f"SELECT `{column}` FROM `{table}` LIMIT 5")
            rows = cur.fetchall()
        except Exception as e2:
            print(f"Error: 连降级查询也失败 for {table}.{column}: {e2}")
            rows = []
    
    return rows

def get_data_from_milvus_sparse(db, table_name, column_name, question, evidence):
    
    COLLECTION_NAME = f"{db}_sparse"
    dense, sparse = get_vector([question, evidence])
    if len(sparse) == 2:
        # 生成查询向量
        question_vector = sparse[0]
        evidence_vector = sparse[1]

        search_param_1 = {
            "data": [question_vector],
            "anns_field": "value_vector",
            "param": {
                "metric_type": "IP",
            },
            "limit": 5,
            "expr": f"table_name == '{table_name}' AND original_column_name == '{column_name}'"
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [evidence_vector],
            "anns_field": "value_vector",
            "param": {
                "metric_type": "IP",
            },
            "limit": 5,
            "expr": f"table_name == '{table_name}' AND original_column_name == '{column_name}'"
        }
        request_2 = AnnSearchRequest(**search_param_2)

        reqs = [request_1, request_2]
    elif len(sparse) == 1:
        # 生成查询向量
        question_vector = sparse[0]

        search_param_1 = {
            "data": [question_vector],
            "anns_field": "value_vector",
            "param": {
                "metric_type": "IP",
            },
            "limit": 5,
            "expr": f"table_name == '{table_name}' AND original_column_name == '{column_name}'"
        }
        request_1 = AnnSearchRequest(**search_param_1)

        reqs = [request_1]

    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 5,
        output_fields = ["table_name", "original_column_name", "value"]
    )
    results = []  # 结构：{ table_name: [(score, column_name), ...], ... }
    for hits in res:
        for hit in hits:
            results.append(hit['entity']['value'])
    # print(f"{table_name}-----{column_name}---------{results}")
    return results

def get_explame(question):
    COLLECTION_NAME = "QA_example"
    dense, sparse = get_vector([question])
    dense = dense[0].tolist() if hasattr(dense[0], 'tolist') else dense[0]

    res = client.search(
        collection_name = COLLECTION_NAME,
        anns_field = "question_vector",
        data = [dense],
        limit = 1,
        search_params={"metric_type": "COSINE"},
        output_fields = ["question", "sql"]
    )
    results = {}
    for hits in res:
        for hit in hits:
            results['question'] = hit['entity']['question']
            results['sql'] = hit['entity']['sql']
    return results

def process_item(item):
    try:
        question = item.get("question")
        evidence = item.get("evidence")
        db = item.get("db")
        columns = item.get("columns")
        
        # 规范化列名
        tables, columns = normalize_column_name(db, columns)
        # print("columns: ", columns)
        # print("tables: ", tables)
        # 完善模式信息和外键信息
        schema, foreign_key, columns = build_schema(db, tables, columns)
        # print("schema: ", schema)
        # print("foreign_key: ", foreign_key)
        # print("columns: ", columns)

        explanation = build_explanation(db, columns)
        # print("explanation: ", explanation)
        # print("explanation: ", explanation)
        data = build_data(db, columns, question, evidence)
        print("data: ", data)
        # print("data: ", data)
        explame = get_explame(question)
        return {
            "question_id": item.get("question_id", ""),
            "db": db,
            "question": question,
            "evidence": evidence,
            "columns": columns,
            "sql_1": item.get("sql_1"),
            "schema": schema,
            "foreign_key": foreign_key,
            "explanation": explanation,
            "explame": explame,
            "data": data,
            "difficulty": item.get("difficulty", "")
        }
    except KeyError as e:
        print(f"Warning: 缺少键 {e}，item={item}")
    except Exception as e:
        print(f"Error processing item: {e}")
    return None

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
        for item in results:
            # 移除 indent 参数，添加换行符
            json_line = json.dumps(
                item,
                ensure_ascii=False,
                default=lambda o: list(o) if isinstance(o, set) else o
            )
            f.write(json_line + '\n')  # 显式添加换行符

def main(input_file, output_file, start_index, max_workers = 1):
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
    parser.add_argument("--start_index", type = int, default = 0)
    parser.add_argument("--input_file", type = str, default = "src/dataset/qwen/coder-32b/en/1_sl_final_coder.jsonl")
    parser.add_argument("--output_file", type = str, default = "src/dataset/qwen/coder-32b/en/1_5_normalize_schema1.jsonl")
    parser.add_argument("--input_file1", type = str, default = "src/dataset/qwen/coder-32b/en/1_sl_final_coder_null.jsonl")
    parser.add_argument("--output_file1", type = str, default = "src/dataset/qwen/coder-32b/en/1_5_normalize_schema_null.jsonl")
    args = parser.parse_args()
    # extract_error_json(args.input_file, args.output_file, args.input_file1)
    main(args.input_file, args.output_file, args.start_index)