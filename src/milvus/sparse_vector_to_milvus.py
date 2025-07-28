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


def get_vector(query):
    """
    生成 embedding 向量并返回 dense 部分列表。
    """
    try:
        vecs = bge_m3_ef.encode_documents(query)

        # 获取稀疏矩阵对象
        sparse_obj = vecs['sparse']
        dok = sparse_obj.todok()

        # 将 DOK 转换为普通字典
        sparse_dict = dict(dok)

        # 初始化字典
        dict_dict = {}

        # 遍历稀疏字典并构建新的字典
        for (row, col), value in sparse_dict.items():
            if row not in dict_dict:
                dict_dict[int(row)] = {}
            dict_dict[int(row)][int(col)] = float(value)

        # 如果需要将结果转换为列表形式
        dict_list = [cols for row, cols in dict_dict.items()]

        return dict_list
        
        # return vecs["sparse"]
        
    except Exception as e:
        print(f"[Error] 生成向量失败，query={query!r}，原因：{e}")
        return None
    
def read_data():
    base_dir = os.path.join(os.getcwd(), 'milvus', 'sparse')
    if not os.path.isdir(base_dir):
        print(f"[Error] 目录不存在：{base_dir}")
        return
    values_dir = os.path.join(base_dir, 'values')
    files = [f for f in os.listdir(values_dir) if os.path.isfile(os.path.join(values_dir, f))]
    if not files:
        print(f"[Warning] 目录 {values_dir} 下没有文件。")
        return

    for fname in tqdm(files, desc="处理文件"):
        path = os.path.join(values_dir, fname)
        name, _ = os.path.splitext(fname)

        try:
            items = json.load(open(path, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"[Error] 读取 {fname} 失败: {e}")
            continue

        results = []
        # 根据顶层类型分支
        
        for rec in tqdm(items, desc=f"{fname} embedding", leave=False):
            tbl = rec.get('table_name')
            col = rec.get('column_name')
            values = rec.get('values')
            values_vector = get_vector(values)
            for value, vector in zip(values, values_vector):
                results.append({
                    "table_name": tbl,
                    "original_column_name": col,
                    "value": value,
                    "value_vector": vector
                })

        # 写 JSON
        out_path = os.path.join(base_dir, f"{name}_to_milvus.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"[OK] 已写入 {out_path} （共 {len(results)} 条）")

        # 插入 Milvus
        sparse_vector_to_milvus(name, results)

def sparse_vector_to_milvus(db_name, data):

    collection_name = f"{db_name}_sparse"

    if not data:
        print(f"[Warning] 无数据可插入 Milvus 集合 {collection_name}")
        return
    
    client = MilvusClient(
        uri = MILVUS_URI, 
        token = MILVUS_TOKEN
    )

    try:
        # 删除已有集合
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"[Milvus] 已删除旧集合 {collection_name}")

        # 构建 schema
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("table_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("original_column_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("value", DataType.VARCHAR, max_length=2048)
        schema.add_field("value_vector", DataType.SPARSE_FLOAT_VECTOR)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name = "value_vector",
            index_name = "value_vector_index",
            index_type = "SPARSE_INVERTED_INDEX",
            metric_type = "IP",
            params = {
                "inverted_index_algo": "DAAT_MAXSCORE"
            }, # or "DAAT_WAND" or "TAAT_NAIVE"
        )

        # 创建集合并建立索引
        client.create_collection(
            collection_name = collection_name,
            schema = schema,
            index_params = index_params
        )
        print(f"[Milvus] 已创建集合 {collection_name}")

        # 批量插入
        client.insert(collection_name = collection_name, data = data)
        client.flush(collection_name)
        print(f"[Milvus] 已插入 {len(data)} 条记录到集合 {collection_name}")
    except Exception as e:
        print(f"[Error] 向 Milvus 插入数据失败: {e}")


if __name__ == "__main__":
    read_data()

    