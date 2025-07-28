import os
import json
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

# # 全局初始化 embedding 函数
# bge_m3_ef = BGEM3EmbeddingFunction(
#     model_name = 'BAAI/bge-m3',
#     device = 'cuda:0',
#     use_fp16 = True
# )

with open('data/QA.json', 'r', encoding='utf-8') as f:
    sql_example = json.load(f)


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

def process_record():
    results = []
    for record in tqdm(sql_example, desc="处理记录", leave=False):
        question = record.get('question', '')
        vectors = get_vector([question])
        if not vectors:
            return None

        entity = {
            "question": question,
            "sql": record.get('sql'),
            "question_vector": vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0]
        }

        results.append(entity)

    # 写 JSON
    out_path = "milvus/example/sql_example.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[OK] 已写入 {out_path} （共 {len(results)} 条）")

    return results

def multi_vector_to_milvus(data):
    
    collection_name = "QA_example"
    if not data:
        print(f"[Warning] 无数据可插入 Milvus 集合 {collection_name}")
        return

    # 取第一条记录的向量维度
    dim = len(data[0]["question_vector"])
    client = MilvusClient(uri=MILVUS_URI, token = MILVUS_TOKEN)

    try:
        # 删除已有集合
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"[Milvus] 已删除旧集合 {collection_name}")

        # 构建 schema
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("sql", DataType.VARCHAR, max_length=2048)

        schema.add_field("question_vector", DataType.FLOAT_VECTOR, dim=dim)

        # 创建集合并建立索引
        client.create_collection(
            collection_name = collection_name,
            schema=schema,
            index_params=[
                {
                    "field_name": "question_vector",
                    "index_type": "FLAT",
                    "metric_type": "COSINE",
                    "index_name": "question_vec"
                }
            ]
        )
        print(f"[Milvus] 已创建集合 {collection_name}，向量维度={dim}")

        total = len(data)
        batch_size = 1000
        for i in range(0, total, batch_size):
            batch = data[i:i + batch_size]
            client.insert(collection_name = collection_name, data = batch)
            print(f"已插入 {min(i + batch_size, total)}/{total} 条数据")
        client.flush(collection_name)
        print(f"[Milvus] 已插入 {len(data)} 条记录到集合 {collection_name}")
    except Exception as e:
        print(f"[Error] 向 Milvus 插入数据失败: {e}")

    

if __name__ == '__main__':
    print("当前工作目录:", os.getcwd())
    # results = process_record()
    out_path = "milvus/example/sql_example.json"
    with open(out_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # 插入 Milvus
    multi_vector_to_milvus(results)
