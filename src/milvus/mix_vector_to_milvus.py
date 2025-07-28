import os
import json
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

# Milvus 连接配置
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"

# 加载结构配置文件
structure_path = os.path.join(os.getcwd(), "milvus", "tables_structure_milvus.json")
if not os.path.exists(structure_path):
    raise FileNotFoundError(f"[Error] 未找到结构定义文件：{structure_path}")
with open(structure_path, 'r', encoding='utf-8') as f:
    table_vector_list = json.load(f)


def multi_vector_to_milvus(db_name, data):
    """
    将处理好的 data 列表插入到 Milvus。
    """
    collection_name = f"{db_name}_mix"
    if not data:
        print(f"[Warning] 无数据可插入 Milvus 集合 {collection_name}")
        return

    dim = len(data[0]["column_name_vector"])
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"[Milvus] 已删除旧集合 {collection_name}")

        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("original_table_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("original_column_name", DataType.VARCHAR, max_length=2048)
        # schema.add_field("table_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("column_name", DataType.VARCHAR, max_length=2048)
        schema.add_field("column_description", DataType.VARCHAR, max_length=2048)
        schema.add_field("value_description", DataType.VARCHAR, max_length=2048)
        # schema.add_field("meaning", DataType.VARCHAR, max_length=2048)
        schema.add_field("value", DataType.VARCHAR, max_length=2048)

        # schema.add_field("table_name_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("column_name_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("column_description_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("value_description_vector", DataType.FLOAT_VECTOR, dim=dim)
        # schema.add_field("meaning_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("value_vector", DataType.SPARSE_FLOAT_VECTOR)

        client.create_collection(
            collection_name=collection_name,
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
                # {
                #     "field_name": "meaning_vector",
                #     "index_type": "FLAT",
                #     "metric_type": "COSINE",
                #     "index_name": "idx_meaning_vec"
                # },
                {
                    "field_name": "value_vector",
                    "index_name": "value_vector_index",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                    "params": {
                        "inverted_index_algo": "DAAT_MAXSCORE"
                    }
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

def build_mix_vector(dense_items, sparse_items, table_name_vectors, mix_path):
    mix_list = []

    for sparse in sparse_items:
        try:
            # 获取并清洗表名和列名
            original_table_name = sparse.get("table_name", "").strip()
            original_column_name = sparse.get("original_column_name", "").strip()

            if not original_table_name or not original_column_name:
                print(f"[Warning] 缺少表或列名，跳过：{sparse}")
                continue

            # 找表结构向量
            # table_vector = next(
            #     (t for t in table_name_vectors if t.get("table_name_original", "").strip() == original_table_name),
            #     None
            # )
            # if not table_vector:
            #     print(f"[Warning] 找不到表结构向量定义：{original_table_name}")
            #     continue

            # 找 dense 向量项
            dense_item = next(
                (
                    d for d in dense_items
                    if d.get("original_table_name", "").strip() == original_table_name
                    and d.get("original_column_name", "").strip() == original_column_name
                ),
                None
            )
            if not dense_item:
                print(f"[Warning] dense_items 中找不到：{original_table_name} - {original_column_name}")
                continue

            # 构建 mix 向量
            mix = {
                "original_table_name": original_table_name,
                "original_column_name": original_column_name,
                # "table_name": table_vector.get("table_name", "").strip(),
                # "table_name_vector": table_vector["table_name_vector"],

                "column_name": dense_item.get("column_name", "").strip(),
                "column_description": dense_item.get("column_description", "").strip(),
                "value_description": dense_item.get("value_description", "").strip(),
                # "meaning": dense_item.get("meaning", "").strip(),
                "value": sparse.get("value", "").strip() if isinstance(sparse.get("value"), str) else sparse.get("value"),

                "column_name_vector": dense_item["column_name_vector"],
                "column_description_vector": dense_item["column_description_vector"],
                "value_description_vector": dense_item["value_description_vector"],
                # "meaning_vector": dense_item["meaning_vector"],
                "value_vector": sparse["value_vector"],
            }

            mix_list.append(mix)

        except Exception as e:
            print(f"[Error] 构建失败（{original_table_name} - {original_column_name}）：{e}")
            continue

    with open(mix_path, "w", encoding="utf-8") as f:
        json.dump(mix_list, f, indent=4, ensure_ascii=False)

    print(f"[OK] 已写入 {mix_path}（共 {len(mix_list)} 条）")
    return mix_list

def read_data():
    dense_dir = os.path.join(os.getcwd(), 'milvus', 'dense', 'vector')
    sparse_dir = os.path.join(os.getcwd(), 'milvus', 'sparse', 'vector')

    files = [f for f in os.listdir(dense_dir) if os.path.isfile(os.path.join(dense_dir, f))]

    for fname in tqdm(files, desc="处理文件"):
        dense_path = os.path.join(dense_dir, fname)
        sparse_path = os.path.join(sparse_dir, fname)

        name, _ = os.path.splitext(fname)
        name_key = name.replace('_to_milvus', '')  # 原始表名
        mix_path = os.path.join(os.getcwd(), 'milvus', 'mix', f'{name}.json')

        try:
            if name_key not in table_vector_list:
                print(f"[Warning] 未在结构定义中找到表 {name_key}，跳过")
                continue

            table_name_vectors = table_vector_list[name_key]
            with open(dense_path, 'r', encoding='utf-8') as f_dense, \
                 open(sparse_path, 'r', encoding='utf-8') as f_sparse:
                dense_items = json.load(f_dense)
                sparse_items = json.load(f_sparse)

            results = build_mix_vector(dense_items, sparse_items, table_name_vectors, mix_path)
            multi_vector_to_milvus(name_key, results)

        except Exception as e:
            print(f"[Error] 读取 {fname} 失败: {e}")


if __name__ == '__main__':
    print("当前工作目录:", os.getcwd())
    read_data()
