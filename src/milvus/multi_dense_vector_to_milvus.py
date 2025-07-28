import os
import json
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

# 全局初始化 embedding 函数
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name = 'BAAI/bge-m3',
    device = 'cuda:0',
    use_fp16 = True
)

with open('data/column_meaning.json', 'r', encoding='utf-8') as f:
    meanings = json.load(f)

with open('milvus/tables_structure_milvus.json', 'r', encoding='utf-8') as f:
    table_structure = json.load(f)
def get_vector(query):
    """
    生成 embedding 向量并返回 dense 部分列表。
    """
    try:
        vecs = bge_m3_ef.encode_documents(query)
        return vecs["dense"]
    except Exception as e:
        print(f"[Error] 生成向量失败，query={query!r}，原因：{e}")
        return None

def process_record(db_name, table_name_org, record):
    """
    单条记录处理：取三段文本生成向量，返回拼装好的 dict。
    """
    col_nm = record.get('column_name', '')
    col_desc = record.get('column_description', '')
    val_desc = record.get('value_description', '')
    # for item in table_structure.get(db_name):
    #     if item['table_name_original'] == table_name_org:
    #         table_name = item['table_name']
    #         table_name_vector = item['table_name_vector']
    #         break
    meaning = meanings.get(f'{db_name}|{table_name_org}|{col_nm}')
    if not meaning:
        meaning = val_desc
    vectors = get_vector([col_nm, col_desc, val_desc, meaning])
    if not vectors or len(vectors) < 3:
        return None

    return {
        "original_table_name": table_name_org,
        # "table_name": table_name,
        "original_column_name": record.get('original_column_name', ''),
        "column_name": col_nm,
        "column_name_vector": vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0],
        "column_description": col_desc,
        "column_description_vector": vectors[1].tolist() if hasattr(vectors[1], 'tolist') else vectors[1],
        "value_description": val_desc,
        "value_description_vector": vectors[2].tolist() if hasattr(vectors[2], 'tolist') else vectors[2],
        "meaning": meaning,
        "meaning_vector": vectors[3].tolist() if hasattr(vectors[3], 'tolist') else vectors[3],
        # "table_name_vector": table_name_vector
    }

def multi_vector_to_milvus(db_name, data):
    """
    将处理好的 data 列表插入到 Milvus：
    - 如果集合已存在则先删除
    - 动态获取向量维度
    - 插入后 flush
    """
    collection_name = f"{db_name}_dense"
    if not data:
        print(f"[Warning] 无数据可插入 Milvus 集合 {collection_name}")
        return

    # 取第一条记录的向量维度
    dim = len(data[0]["column_name_vector"])
    client = MilvusClient(uri=MILVUS_URI, token = MILVUS_TOKEN)

    try:
        # 删除已有集合
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"[Milvus] 已删除旧集合 {collection_name}")

        # 构建 schema
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("original_table_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("original_column_name", DataType.VARCHAR, max_length=2048)
        # schema.add_field("table_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("column_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("column_description", DataType.VARCHAR, max_length=2048)
        schema.add_field("value_description", DataType.VARCHAR, max_length=2048)
        schema.add_field("meaning", DataType.VARCHAR, max_length=2048)
        # schema.add_field("table_name_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("column_name_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("column_description_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("value_description_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("meaning_vector", DataType.FLOAT_VECTOR, dim=dim)

        # 创建集合并建立索引
        client.create_collection(
            collection_name = collection_name,
            schema=schema,
            index_params=[
                {
                    "field_name": "column_name_vector",
                    "index_type": "FLAT",
                    "metric_type": "COSINE",
                    "index_name": "idx_col_name_vec"
                },
                {
                    "field_name": "column_description_vector",
                    "index_type": "FLAT",
                    "metric_type": "COSINE",
                    "index_name": "idx_col_desc_vec"
                },
                {
                    "field_name": "value_description_vector",
                    "index_type": "FLAT",
                    "metric_type": "COSINE",
                    "index_name": "idx_val_desc_vec"
                },
                {
                    "field_name": "meaning_vector",
                    "index_type": "FLAT",
                    "metric_type": "COSINE",
                    "index_name": "idx_meaning_vec"
                }
            ]
        )
        print(f"[Milvus] 已创建集合 {collection_name}，向量维度={dim}")

        # 批量插入
        client.insert(collection_name = collection_name, data = data)
        client.flush(collection_name)
        print(f"[Milvus] 已插入 {len(data)} 条记录到集合 {collection_name}")
    except Exception as e:
        print(f"[Error] 向 Milvus 插入数据失败: {e}")

def read_data():
    """
    遍历 milvus/dense 下所有 JSON 文件，生成向量并写入新 JSON，同时插入 Milvus。
    """
    base_dir = os.path.join(os.getcwd(), 'milvus', 'dense')
    if not os.path.isdir(base_dir):
        print(f"[Error] 目录不存在：{base_dir}")
        return

    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    if not files:
        print(f"[Warning] 目录 {base_dir} 下没有文件。")
        return

    for fname in tqdm(files, desc="处理文件"):
        path = os.path.join(base_dir, fname)
        name, _ = os.path.splitext(fname)

        try:
            items = json.load(open(path, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"[Error] 读取 {fname} 失败: {e}")
            continue

        results = []
        # 根据顶层类型分支
        if isinstance(items, dict):
            for table, recs in items.items():
                if not isinstance(recs, list):
                    print(f"[Warning] {fname} 中 {table} 不是列表，已跳过")
                    continue
                for rec in tqdm(recs, desc=f"{fname} → {table}", leave=False):
                    pr = process_record(name, table, rec)
                    if pr: results.append(pr)

        elif isinstance(items, list):
            for rec in tqdm(items, desc=f"{fname} → 列表记录", leave=False):
                tbl = rec.get('table', '')
                pr = process_record(name, tbl, rec)
                if pr: results.append(pr)

        else:
            print(f"[Warning] 无法识别 {fname} 顶层类型 {type(items)}, 已跳过")
            continue

        # 写 JSON
        out_path = os.path.join(base_dir, f"{name}_to_milvus.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"[OK] 已写入 {out_path} （共 {len(results)} 条）")

        # 插入 Milvus
        multi_vector_to_milvus(name, results)

if __name__ == '__main__':
    print("当前工作目录:", os.getcwd())
    read_data()
