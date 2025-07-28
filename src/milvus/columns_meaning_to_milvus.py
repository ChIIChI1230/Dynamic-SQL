import json
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "bird_columns_search_by_meaning"

# 连接 Milvus
client = MilvusClient(
    uri = "http://localhost:19530",
    token = "root:Milvus"
)
client.drop_collection(
    collection_name = COLLECTION_NAME
)
# 定义字段
schema = client.create_schema(
    auto_id = True,
    enable_dynamic_field = True
)


schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="table_name_original", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="column_name_original", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="column_name_meaning", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="column_name_meaning_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="column_name_meaning_vector",
    index_type="FLAT",
    index_name="column_name_meaning_vector_index",
    metric_type="L2",
    params={}
)

# 创建集合
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)


data = {}
with open("./milvus/columns_meaning_milvus.json", "r", encoding = "utf-8") as f:
    data = json.load(f)
print(len(data))

# # 加载集合
# client.load_collection(
#     collection_name = COLLECTION_NAME
# )

for db_id, columns_list in tqdm(data.items(), desc="Creating partitions"):

    # 创建分区
    client.create_partition(
        collection_name = COLLECTION_NAME,
        partition_name = db_id
    )

    client.insert(
        collection_name = COLLECTION_NAME,
        partition_name = db_id,
        data = columns_list
    )

    # client.flush(
    #     collection_name = COLLECTION_NAME
    # )

# # 释放集合，将所有的数据全部删除
# client.release_collection(
#     collection_name = COLLECTION_NAME
# )