import json
import sqlite3
import random
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from collections import Counter, defaultdict
from typing import List, Dict



def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_vector(query):
    # 生成 embedding 向量
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name = 'BAAI/bge-m3',  # Specify the model name
        device = 'cpu',              # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16 = False             # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )
    query_vector_list = bge_m3_ef.encode_documents([query])
    dense_vector = query_vector_list["dense"]    
    return dense_vector

# 从指定 SQLite 数据库中查询给定表的指定列的所有非空值并去重
def get_unique_text_values(db_id, table_name, column_name):
    db_file = f"database/dev_databases/{db_id}/{db_id}.sqlite"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL"

    try:
        cur.execute(query)
        results = cur.fetchall()
        values = [row[0] for row in results if row[0] is not None]
    except Exception as e:
        print(f"Error querying {table_name}.{column_name} in {db_file}: {e}")
        values = []
    finally:
        cur.close()
        conn.close()
    return values

def build_structure(dev_tables, column_meanings):
    result = {}
    for db in dev_tables:
        db_id = db.get("db_id")
        # 获取原始和更新后的表名
        table_names_ori = db.get("table_names_original", [])
        table_names_upd = db.get("table_names", [])
        # 获取列信息（原始和更新后）
        columns_ori = db.get("column_names_original", [])
        columns_upd = db.get("column_names", [])
        
        # 初始化 db 结构
        result[db_id] = {
            "tables": {},  # 存储原始表名到更新后表名的映射
            "colums": {}   # 存储每个表中列信息的映射
        }
        
        # 构建表名映射： key 为原始表名， value 为更新后的表名
        for idx, table_ori in enumerate(table_names_ori):
            if idx < len(table_names_upd):
                result[db_id]["tables"][table_ori] = table_names_upd[idx]
            else:
                result[db_id]["tables"][table_ori] = ""
            # 初始化 colums 下对应表的结构
            result[db_id]["colums"][table_ori] = {}
        
        # 遍历列信息，两份列表对应位置相同
        for (col_ori, col_upd) in zip(columns_ori, columns_upd):
            table_index, col_name_ori = col_ori
            _, col_name_upd = col_upd
            # 跳过 table_index 为 -1 的项（通常表示 '*'）
            if table_index == -1:
                continue
            # 如果 table_index 超出范围则跳过
            if table_index >= len(table_names_ori):
                continue
            table_ori = table_names_ori[table_index]
            # 构造拼接键，格式： "db_id|table_names_original|column_names_original"
            key = f"{db_id}|{table_ori}|{col_name_ori}"
            description = column_meanings.get(key, "").lstrip("#").strip()
            # 保存列信息：键为原始列名，值为 [更新后的列名, 描述]
            result[db_id]["colums"][table_ori][col_name_ori] = [col_name_upd, description]
            
        # 如果需要，可以在此处对结果进行进一步处理
    return result

def build_table_structure(dev_tables):
    # 生成 embedding 向量
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',  # Specify the model name
        device='cpu',              # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False             # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )
    result = {}
    # 使用 tqdm 包装迭代器，显示进度条
    for db in tqdm(dev_tables, desc="Processing databases"):
        db_id = db.get("db_id")
        table_names_ori = db.get("table_names_original", [])
        table_names_upd = db.get("table_names", [])
        table_names_upd_vector = bge_m3_ef.encode_documents(table_names_upd)
        dense_vector = table_names_upd_vector["dense"]

        # 构造表的列表，每个表为一个字典，包含三个字段
        tables_list = []
        for idx, table_ori in enumerate(table_names_ori):
            table_entry = {
                "table_name_original": table_ori,
                "table_name": table_names_upd[idx],
                "table_name_vector": dense_vector[idx].tolist()  # 转换为列表形式
            }
            tables_list.append(table_entry)
        
        result[db_id] = tables_list
    return result

def build_column_structure(dev_tables):
    # 生成 embedding 向量
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',  # Specify the model name
        device='cpu',              # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False             # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )

    result = {}
    for db in tqdm(dev_tables, desc = "Processing databases"):
        db_id = db.get("db_id")
        columns_list = []
        # 获得表名
        table_names_ori = db.get("table_names_original", [])
        # 获得列名
        columns_ori = db.get("column_names_original", [])
        columns_upd = db.get("column_names", [])

        # 遍历列信息，两份列表对应位置相同
        for (col_ori, col_upd) in zip(columns_ori, columns_upd):
            table_index, col_name_ori = col_ori
            _, col_name_upd = col_upd
            # 跳过 table_index 为 -1 的项（通常表示 '*'）
            if table_index == -1:
                continue
            # 如果 table_index 超出范围则跳过
            if table_index >= len(table_names_ori):
                continue
            table_ori = table_names_ori[table_index]
            
            column_name_vector = bge_m3_ef.encode_documents([col_name_upd])
            dense_vectors = column_name_vector["dense"]
            column_name_vector = dense_vectors[0].tolist() if hasattr(dense_vectors[0], "tolist") else list(dense_vectors[0])
            
            column_entry = {
                "table_name_original": table_ori,
                "column_name_original": col_name_ori,
                "column_name": col_name_upd,
                "column_name_vector": column_name_vector
            }
            columns_list.append(column_entry)
        
        result[db_id] = columns_list
    
    return result

def build_column_meaning(column_meanings):
    
    result = {}
    for key, meaning in tqdm(column_meanings.items()):
        parts = key.split("|")
        db_id = parts[0]

        meaning_vector = get_vector(meaning)
        meaning_vector1 = meaning_vector[0].tolist() if hasattr(meaning_vector[0], "tolist") else list(meaning_vector[0])
        entry = {
            "table_name_original": parts[1],
            "column_name_original": parts[2],
            "column_name_meaning": meaning,
            "column_name_meaning_vector": meaning_vector1
        }

        if db_id in result:
            result[db_id].append(entry)
        else:
            result[db_id] = [entry]

    return result

def build_column_values(dev_tables):
    # 生成 embedding 向量
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',  # Specify the model name
        device='cpu',              # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False             # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )

    result = {}
    for db in tqdm(dev_tables, desc = "Processing databases"):
        db_id = db.get("db_id")
        column_value_list = []
        # 获得表名
        table_names_ori = db.get("table_names_original", [])
        # 获得列名
        columns_ori = db.get("column_names_original", [])

        column_types = db.get("column_types", [])

        for (col_ori, type) in tqdm(zip(columns_ori, column_types),
                                        desc=f"Processing columns in {db_id}",
                                        leave=False):
            
            table_index, col_name_ori = col_ori
            
            # 跳过 table_index 为 -1 的项（通常表示 '*'）
            if table_index == -1:
                continue
            # 如果 table_index 超出范围则跳过
            if table_index >= len(table_names_ori):
                continue
            
            table_name_ori = table_names_ori[table_index]

            if type == "text":
                values = get_unique_text_values(db_id, table_name_ori, col_name_ori)
                # values_vector = bge_m3_ef.encode_documents(values)
                # dense_vector = values_vector["dense"]

                for i, value in enumerate(values):
                    column_value_list.append({
                        "table_name_original": table_name_ori,
                        "column_name_original": col_name_ori,
                        "column_value": value,
                        # "column_value_vector": dense_vector[i]
                        "column_value_vector": ""
                    })
        
        result[db_id] = column_value_list
    return result

def count_column_values(dev_tables):
    result = {}
    for db in tqdm(dev_tables, desc = "Processing databases"):
        db_id = db.get("db_id")
        column_value_list = []
        # 获得表名
        table_names_ori = db.get("table_names_original", [])
        
        # 获得列名
        columns_ori = db.get("column_names_original", [])

        column_types = db.get("column_types", [])

        count = 0

        for (col_ori, type) in tqdm(zip(columns_ori, column_types),
                                        desc=f"Processing columns in {db_id}",
                                        leave=False):
            
            table_index, col_name_ori = col_ori
            
            # 跳过 table_index 为 -1 的项（通常表示 '*'）
            if table_index == -1:
                continue
            # 如果 table_index 超出范围则跳过
            if table_index >= len(table_names_ori):
                continue
            
            table_name_ori = table_names_ori[table_index]

            if type == "text":
                values = get_unique_text_values(db_id, table_name_ori, col_name_ori)
                # values_vector = bge_m3_ef.encode_documents(values)
                # dense_vector = values_vector["dense"]
                count += len(values)
                column_value_list.append({
                        "table_name_original": table_name_ori,
                        "column_name_original": col_name_ori,
                        "count_column_value": len(values),
                        "all": count
                    })

        
        result[db_id] = column_value_list
    return result



# 向量空间构建工具类
def build_table_vectors(db) -> List[Dict]:
    """构建表向量空间"""
    result = []
    table_names = db.get("table_names_original", [])
    table_comments = db.get("table_comments", ["" for _ in table_names])

    for table_name, comment in zip(table_names, table_comments):
        text = f"表名：{table_name}，描述：{comment}"
        result.append({
            "type": "table",
            "table_name": table_name,
            "text": text
        })
    return result

def build_column_vectors(db, max_values=5) -> List[Dict]:
    """构建列解释向量空间"""
    result = []
    table_names = db.get("table_names_original", [])
    columns = db.get("column_names_original", [])
    types = db.get("column_types", [])

    for (table_idx, column_name), column_type in zip(columns, types):
        if table_idx == -1:
            continue
        table_name = table_names[table_idx]
        sample_values = get_unique_text_values(db["db_id"], table_name, column_name)[:max_values]
        text = f"{table_name} 表的字段 {column_name}（类型：{column_type}），例如：{'、'.join(sample_values)}"
        result.append({
            "type": "column",
            "table_name": table_name,
            "column_name": column_name,
            "text": text
        })
    return result

def build_value_vectors(db, max_values_per_column=50, strategy="topN") -> List[Dict]:
    """构建字段值向量空间"""
    result = []
    table_names = db.get("table_names_original", [])
    columns = db.get("column_names_original", [])
    types = db.get("column_types", [])

    for (table_idx, column_name), column_type in zip(columns, types):
        if column_type != "text" or table_idx == -1:
            continue
        table_name = table_names[table_idx]
        values = get_unique_text_values(db["db_id"], table_name, column_name)

        if len(values) > max_values_per_column:
            if strategy == "topN":
                values = [val for val, _ in Counter(values).most_common(max_values_per_column)]
            elif strategy == "random":
                values = random.sample(values, max_values_per_column)

        for val in values:
            result.append({
                "type": "value",
                "table_name": table_name,
                "column_name": column_name,
                "value": val,
                "text": val
            })
    return result

def embed_all(entries: List[Dict], embedder: EmbeddingFunction) -> List[Dict]:
    texts = [entry["text"] for entry in entries]
    vectors = embedder.encode(texts)
    for entry, vec in zip(entries, vectors):
        entry["vector"] = vec
    return entries

# Schema linking 检索结果展示结构
def schema_linking_demo(user_query_vector, vector_db, top_k=3):
    """
    user_query_vector: 用户问题的向量
    vector_db: 所有嵌入项（表、列、字段值）
    """
    # 用余弦相似度排序（简单写法）
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    candidates = []
    for item in vector_db:
        sim = cosine_similarity([user_query_vector], [item["vector"]])[0][0]
        candidates.append((sim, item))

    top_items = sorted(candidates, key=lambda x: -x[0])[:top_k]
    return [item for _, item in top_items]

# 查询日志驱动的向量库动态更新机制
def update_value_vectors_from_logs(query_logs: List[str], existing_value_index: Dict[str, set], db_info, embedder):
    """
    query_logs: 用户历史问题集合
    existing_value_index: 已有值的索引，用于去重，如 {(table, column): set(values)}
    db_info: 数据库结构（含表名、列名）
    """
    new_entries = []
    for query in query_logs:
        for table in db_info.get("table_names_original", []):
            for (table_idx, col_name), col_type in zip(db_info["column_names_original"], db_info["column_types"]):
                if col_type != "text" or table_idx == -1:
                    continue
                full_col = (table, col_name)
                # 简单匹配：列名或已有值是否出现在 query 中
                for val in get_unique_text_values(db_info["db_id"], table, col_name):
                    if val in query and val not in existing_value_index.get(full_col, set()):
                        new_entries.append({
                            "type": "value",
                            "table_name": table,
                            "column_name": col_name,
                            "value": val,
                            "text": val
                        })
                        existing_value_index.setdefault(full_col, set()).add(val)

    return embed_all(new_entries, embedder)


def main():
    # 加载文件
    dev_tables = load_json("D:\\code\\vs_code\\synthlink-sql\\data\\dev_tables.json")
    column_meanings = load_json("D:\\code\\vs_code\\synthlink-sql\\data\\column_meaning.json")
    
    # 提取表和列的描述信息
    # extracted = build_structure(dev_tables, column_meanings)
    
    # # 保存为 JSON 文件
    # output_file = ".\\milvus\\bird_db_structure.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(extracted, f, indent=2, ensure_ascii=False)
    
    # print(f"提取结果已保存至 {output_file}")

    # tables_structure = build_table_structure(dev_tables)

    # # 保存结果到 JSON 文件
    # output_file = ".\\milvus\\tables_structure_milvus.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(tables_structure, f, indent=2, ensure_ascii=False)
    
    # print(f"提取结果已保存至 {output_file}")

    # columns_structure = build_column_structure(dev_tables)

    # # 保存结果到 JSON 文件
    # output_file = ".\\milvus\\columns_structure_annotations_milvus.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(columns_structure, f, indent=2, ensure_ascii=False)
    
    # print(f"提取结果已保存至 {output_file}")

    # columns_structure_by_meaning = build_column_meaning(column_meanings)

    # # 保存结果到 JSON 文件
    # output_file = ".\\milvus\\columns_meaning_milvus.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(columns_structure_by_meaning, f, indent=2, ensure_ascii=False)
    
    # print(f"提取结果已保存至 {output_file}")

    columns_structure_by_values = count_column_values(dev_tables)

    # 保存结果到 JSON 文件
    output_file = ".\\milvus\\columns_values_milvus_count.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(columns_structure_by_values, f, indent=2, ensure_ascii=False)
    
    print(f"提取结果已保存至 {output_file}")


if __name__ == "__main__":
    main()
