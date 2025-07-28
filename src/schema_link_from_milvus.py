import re
import os
import sys
import argparse
import json
from tqdm import tqdm
from pymilvus import RRFRanker, WeightedRanker
from pymilvus import AnnSearchRequest
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

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

def get_vector(query):
    """
    生成 embedding 向量并返回 dense 部分列表。
    """
    try:
        vecs = bge_m3_ef.encode_documents(query)
        dense = vecs["dense"]
        
        # 获取稀疏矩阵对象
        sparse_obj = vecs['sparse']
        dok = sparse_obj.todok()

        # 将 DOK 转换为普通字典
        sparse_dict = dict(dok)
        dict_dict = {}

        # 遍历稀疏字典并构建新的字典
        for (row, col), value in sparse_dict.items():
            if row not in dict_dict:
                dict_dict[int(row)] = {}
            dict_dict[int(row)][int(col)] = float(value)

        # 如果需要将结果转换为列表形式
        sparse = [cols for row, cols in dict_dict.items()]

        return dense, sparse
    except Exception as e:
        print(f"[Error] 生成向量失败，query={query!r}，原因：{e}")
        return None
    
def match_columns_from_dense_vector1(db_id, vectors):
    
    COLLECTION_NAME = f"{db_id}_dense"
    if len(vectors) == 2:
        # 生成查询向量
        question_vector = vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0]
        evidence_vector = vectors[1].tolist() if hasattr(vectors[1], 'tolist') else vectors[1]

        # print("Query vector:", type(query_vector))
        # 创建8个 AnnSearchRequest 实例
        search_param_1 = {
            "data": [question_vector],
            "anns_field": "column_name_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [question_vector],
            "anns_field": "column_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_2 = AnnSearchRequest(**search_param_2)

        search_param_3 = {
            "data": [question_vector],
            "anns_field": "value_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_3 = AnnSearchRequest(**search_param_3)

        search_param_4 = {
            "data": [question_vector],
            "anns_field": "meaning_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_4 = AnnSearchRequest(**search_param_4)

        search_param_5 = {
            "data": [evidence_vector],
            "anns_field": "column_name_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_5 = AnnSearchRequest(**search_param_5)

        search_param_6 = {
            "data": [evidence_vector],
            "anns_field": "column_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_6 = AnnSearchRequest(**search_param_6)

        search_param_7 = {
            "data": [evidence_vector],
            "anns_field": "value_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_7 = AnnSearchRequest(**search_param_7)

        search_param_8 = {
            "data": [evidence_vector],
            "anns_field": "meaning_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_8 = AnnSearchRequest(**search_param_8)
        reqs = [request_1, request_2, request_3, request_4, request_5, request_6, request_7, request_8]
    elif len(vectors) == 1:
        # 生成查询向量
        question_vector = vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0]
        evidence_vector = vectors[1].tolist() if hasattr(vectors[1], 'tolist') else vectors[1]

        # print("Query vector:", type(query_vector))
        # 创建8个 AnnSearchRequest 实例
        search_param_1 = {
            "data": [question_vector],
            "anns_field": "column_name_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [question_vector],
            "anns_field": "column_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_2 = AnnSearchRequest(**search_param_2)

        search_param_3 = {
            "data": [question_vector],
            "anns_field": "value_description_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_3 = AnnSearchRequest(**search_param_3)

        search_param_4 = {
            "data": [question_vector],
            "anns_field": "meaning_vector",
            "param": {
                "metric_type": "COSINE",
                "params": {}
            },
            "limit": 5
        }
        request_4 = AnnSearchRequest(**search_param_4)
        reqs = [request_1, request_2, request_3, request_4]

    # 配置 Rerankers 策略
    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 10,
        output_fields = ["original_table_name", "original_column_name"]
    )
    results = {}  # 结构：{ table_name: [(score, column_name), ...], ... }
    for hits in res:
        for hit in hits:
            table_name = hit['entity']['original_table_name']
            column_name = hit['entity']['original_column_name']
            score = hit.get("score", 0)
            results.setdefault(table_name, []).append((score, column_name))

    most_relevant_columns = {}
    for table_name, scored_columns in results.items():
        sorted_columns = sorted(scored_columns, key=lambda x: x[0], reverse=True)
        # 只选择前 top_n 个
        top_columns = [col for _, col in sorted_columns][:6]
        most_relevant_columns[table_name] = top_columns

    return most_relevant_columns

def match_columns_from_dense_vector(db_id, vectors):
    
    COLLECTION_NAME = f"{db_id}_dense"
    
    vector_list = []
    for i in range(len(vectors)):
        vector_list.append(vectors[i].tolist() if hasattr(vectors[i], 'tolist') else vectors[i])
    # print("Query vector:", type(query_vector))
    # 创建8个 AnnSearchRequest 实例
    search_param_1 = {
        "data": vector_list,
        "anns_field": "column_name_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 5
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": vector_list,
        "anns_field": "column_description_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 5
    }
    request_2 = AnnSearchRequest(**search_param_2)

    search_param_3 = {
        "data": vector_list,
        "anns_field": "value_description_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 5
    }
    request_3 = AnnSearchRequest(**search_param_3)

    search_param_4 = {
        "data": vector_list,
        "anns_field": "meaning_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 5
    }
    request_4 = AnnSearchRequest(**search_param_4)
    reqs = [request_1, request_2, request_3, request_4]

    # 配置 Rerankers 策略
    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 10,
        output_fields = ["original_table_name", "original_column_name"]
    )
    print("dense:"+ str(res))
    # results = {}  # 结构：{ table_name: [(score, column_name), ...], ... }
    # for hits in res[:10]:
    #     for hit in hits:
    #         table_name = hit['entity']['original_table_name']
    #         column_name = hit['entity']['original_column_name']
    #         score = hit.get("score", 0)
    #         results.setdefault(table_name, []).append((score, column_name))

    # # most_relevant_columns = {}
    # # for table_name, scored_columns in results.items():
    # #     sorted_columns = sorted(scored_columns, key=lambda x: x[0], reverse=True)
    # #     # 只选择前 top_n 个
    # #     top_columns = [col for _, col in sorted_columns][:2]
    # #     most_relevant_columns[table_name] = top_columns
    # # return most_relevant_columns
    # return results
    top_k = 10
    all_scored_columns = []
    all_scored_columns = []
    for hits in res:
        for hit in hits:
            table = hit['entity']['original_table_name']
            column = hit['entity']['original_column_name']
            score = hit.get("score", 0)
            all_scored_columns.append((score, table, column))

    # 保留得分最高的同名字段
    best_column_map = {}  # key: column_name, value: (score, table)
    for score, table, column in all_scored_columns:
        if column not in best_column_map or score > best_column_map[column][0]:
            best_column_map[column] = (score, table)

    # 取前 top_k 个按得分排序
    top_k_columns = sorted(best_column_map.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

    # 构造结果 {table: [column]}
    results = {}
    for column, (score, table) in top_k_columns:
        results.setdefault(table, []).append(column)

    return results

def match_columns_from_sparse_vector1(db_id, sparse):
    COLLECTION_NAME = f"{db_id}_sparse"

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
            "limit": 5
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [evidence_vector],
            "anns_field": "value_vector",
            "param": {
                "metric_type": "IP",
            },
            "limit": 5
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
            "limit": 5
        }
        request_1 = AnnSearchRequest(**search_param_1)

        reqs = [request_1]

    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 5,
        output_fields = ["table_name", "original_column_name"]
    )
    results = {}  # 结构：{ table_name: [(score, column_name), ...], ... }
    for hits in res:
        for hit in hits:
            table_name = hit['entity']['table_name']
            column_name = hit['entity']['original_column_name']
            score = hit.get("score", 0)
            results.setdefault(table_name, []).append((score, column_name))

    most_relevant_columns = {}
    for table_name, scored_columns in results.items():
        sorted_columns = sorted(scored_columns, key=lambda x: x[0], reverse=True)
        # 只选择前 top_n 个
        top_columns = [col for _, col in sorted_columns][:3]
        most_relevant_columns[table_name] = top_columns
    return most_relevant_columns

def match_columns_from_sparse_vector(db_id, sparse):
    COLLECTION_NAME = f"{db_id}_sparse"

    search_param_1 = {
        "data": sparse,
        "anns_field": "value_vector",
        "param": {
            "metric_type": "IP",
        },
        "limit": 5
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": sparse,
        "anns_field": "value_vector",
        "param": {
            "metric_type": "IP",
        },
        "limit": 5
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]
    
    ranker = RRFRanker(100)

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 5,
        output_fields = ["table_name", "original_column_name"]
    )
    print("sparse:" + str(res))
    # results = {}  # 结构：{ table_name: [(score, column_name), ...], ... }
    # for hits in res:
    #     for hit in hits:
    #         table_name = hit['entity']['table_name']
    #         column_name = hit['entity']['original_column_name']
    #         score = hit.get("score", 0)
    #         results.setdefault(table_name, []).append((score, column_name))

    # most_relevant_columns = {}
    # for table_name, scored_columns in results.items():
    #     sorted_columns = sorted(scored_columns, key=lambda x: x[0], reverse=True)
    #     # 只选择前 top_n 个
    #     top_columns = [col for _, col in sorted_columns][:2]
    #     most_relevant_columns[table_name] = top_columns
    # return most_relevant_columns
    top_k = 10
    all_scored_columns = []
    for hits in res:
        for hit in hits:
            table = hit['entity']['table_name']
            column = hit['entity']['original_column_name']
            score = hit.get("score", 0)
            all_scored_columns.append((score, table, column))

    # 保留得分最高的同名字段
    best_column_map = {}  # key: column_name, value: (score, table)
    for score, table, column in all_scored_columns:
        if column not in best_column_map or score > best_column_map[column][0]:
            best_column_map[column] = (score, table)

    # 取前 top_k 个按得分排序
    top_k_columns = sorted(best_column_map.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

    # 构造结果 {table: [column]}
    results = {}
    for column, (score, table) in top_k_columns:
        results.setdefault(table, []).append(column)

    return results
    
def match_columns_tables_from_mix(db_id, dense, sparse):
    COLLECTION_NAME = f"{db_id}_mix"

    dense_list = []
    for i in range(len(dense)):
        dense_list.append(dense[i].tolist() if hasattr(dense[i], 'tolist') else dense[i])
    if len(dense_list) != len(sparse):
        raise ValueError(f"Mismatch between dense ({len(dense_list)}) and sparse ({len(sparse)}) vectors!")
    search_param_1 = {
        "data": dense_list,
        "anns_field": "column_name_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 10
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": dense_list,
        "anns_field": "column_description_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 10
    }
    request_2 = AnnSearchRequest(**search_param_2)

    search_param_3 = {
        "data": dense_list,
        "anns_field": "value_description_vector",
        "param": {
            "metric_type": "COSINE",
            "params": {}
        },
        "limit": 10
    }
    request_3 = AnnSearchRequest(**search_param_3)

    search_param_4 = {
        "data": sparse,
        "anns_field": "value_vector",
        "param": {
            "metric_type": "IP"
        },
        "limit": 10
    }
    request_4 = AnnSearchRequest(**search_param_4)

    reqs = [request_1, request_2, request_3, request_4]

    # 配置 Rerankers 策略
    # ranker = RRFRanker(100)
    ranker = WeightedRanker(0.8, 0.8, 0.8, 0.3) 

    res = client.hybrid_search(
        collection_name = COLLECTION_NAME,
        reqs = reqs,
        ranker = ranker,
        limit = 20,
        output_fields = ["original_table_name", "original_column_name"]
    )
    # print("mix:"+ str(res))
    top_k = 15
    all_scored_columns = []
    for hits in res:
        for hit in hits:
            table = hit['entity']['original_table_name']
            column = hit['entity']['original_column_name']
            score = hit.get("score", 0)
            all_scored_columns.append((score, table, column))

    # 保留得分最高的同名字段
    best_column_map = {}  # key: column_name, value: (score, table)
    for score, table, column in all_scored_columns:
        if column not in best_column_map or score > best_column_map[column][0]:
            best_column_map[column] = (score, table)

    # 取前 top_k 个按得分排序
    top_k_columns = sorted(best_column_map.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

    # 构造结果 {table: [column]}
    results = {}
    for column, (score, table) in top_k_columns:
        results.setdefault(table, []).append(column)

    return results
def match_table_name(db_id, vectors):
    COLLECTION_NAME = "bird_tables_search"
    results = []
    if len(vectors) == 2:
        # 生成查询向量
        question_vector = vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0]
        evidence_vector = vectors[1].tolist() if hasattr(vectors[1], 'tolist') else vectors[1]
        results = client.search(
            collection_name = COLLECTION_NAME,
            partition_names = [db_id],
            data = [question_vector, evidence_vector],  # 单个查询向量
            anns_field = "table_name_vector",
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            },
            limit = 3,
            output_fields = ["table_name_original"]
        )

    elif len(vectors) == 1:
        question_vector = vectors[0].tolist() if hasattr(vectors[0], 'tolist') else vectors[0]
        results = client.search(
            collection_name = COLLECTION_NAME,
            partition_names = [db_id],
            data = [question_vector],  # 单个查询向量
            anns_field = "table_name_vector",
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            },
            limit = 3,
            output_fields = ["table_name_original"]
        )

    table_name_set = set()
    for hits in results:
        # print("TopK results:")
        for hit in hits:
            # print(f"dense:{hit}")
            table_name = hit['entity']['table_name_original']
            table_name_set.add(table_name)
    
    return table_name_set

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

def main(ppl_file, sl_out_file, start_index):
    # with open(ppl_file, 'r', encoding='utf-8') as f:
    #     ppl_data = json.load(f)

    ppl_data = []
    try:
        with open(ppl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ppl_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解码 JSON 行时错误: {e}")
    except Exception as e:
        print(f"读取文件 {ppl_file} 异常: {e}")
        return

    schema_linking_results = []
    for ppl in tqdm(ppl_data[start_index:], desc="Processing PPL"):
        db_id = ppl['db']
        question = ppl['question']
        evidence = ppl['evidence']
        foreign_key = ppl['foreign_key']
        semantic_segmentation_res_list = ppl['semantic_seg_list']
        semantic_segmentation_res_list.append(question)
        if evidence:
            semantic_segmentation_res_list.append(evidence)
        # print(semantic_segmentation_res_list)
        dense, sparse = get_vector(semantic_segmentation_res_list)

        # matched_columns_from_dense = match_columns_from_dense_vector(db_id, dense)

        # matched_columns_from_sparse = match_columns_from_sparse_vector(db_id, sparse)
        
        # 合并 dense 和 sparse 查询结果
        # matched_columns = {}
        # for table_name, column_names in matched_columns_from_dense.items():
        #     matched_columns.setdefault(table_name, set()).update(column_names)
        # for table_name, column_names in matched_columns_from_sparse.items():
        #     matched_columns.setdefault(table_name, set()).update(column_names)

        matched_columns = match_columns_tables_from_mix(db_id, dense, sparse)

        # 构造最终结果中表和列的列表
        tables = match_table_name(db_id, dense)
        columns = []
        for tab, cols in matched_columns.items():
            tables.add(tab)
            for col in cols:
                columns.append(f"{tab}.{col}")
        
        tables = list(tables)
        # 检查外键
        tables_1, columns_1 = prefect_foreign_key(tables, columns, foreign_key)

        # print("Combined matched columns:", matched_columns)
        # print("Tables:", tables)
        # print("Columns:", columns)

        entity = {
            "question_id": ppl['question_id'],
            "db": ppl['db'],
            "question": ppl['question'],
            "evidence": ppl['evidence'],
            "foreign_key": ppl['foreign_key'],
            "matched": matched_columns,
            "tables": tables,
            "columns": columns,
            "tables_1": tables_1,
            "columns_1": columns_1,
            "difficulty": ppl['difficulty']
        }

        schema_linking_results.append(entity)

    try:
        with open(sl_out_file, 'w', encoding='utf-8') as f:
            json.dump(schema_linking_results, f, ensure_ascii=False, indent=4,
                      default=lambda o: list(o) if isinstance(o, set) else o)
        print(f"[OK] 结果已写入 {sl_out_file}")
    except Exception as e:
        print(f"[Error] 写入 {sl_out_file} 失败: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 命令行参数解析
    parser.add_argument("--start_index", type = int, default = 0)
    parser.add_argument("--ppl_file", type = str, default = "src/dataset/qwen/coder-32b/semantic_seg.jsonl")
    parser.add_argument("--sl_out_file", type = str, default = "src/dataset/qwen/coder-32b/sl_out_milvus_sem3.json")
    args = parser.parse_args()

    main(args.ppl_file, args.sl_out_file, args.start_index)