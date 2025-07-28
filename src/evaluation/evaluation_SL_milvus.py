import os
import sys
import json
import copy

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from utils import get_all_schema, extract_tables_and_columns
from config import DEV

def recall_get_column(json_file, output_json_file='extracted_columns.json'):
    """
        计算 SQL 语句的 Schema 预测结果，并评估预测效果
        - 存储 `ground_truths` (真实值)
        - 存储 `columns` (预测值)
        - 计算评估指标（SRR, Avg.T, Avg.C, NSR）
    """
    # 存储每个 gold sql 中涉及的所有 table_name.column_name
    stats = []
    stats_1 = []
    db_schema_copy = copy.deepcopy(get_all_schema())

    with open(DEV.dev_json_path, 'r', encoding='utf-8') as f:
        dev_set = json.load(f)

    # 提取 ground_truths
    ground_truths = []
    for example in dev_set:
        ground_truth = []
        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        stats_1.append(len(ans['table']))  # 记录 SQL 语句涉及的表数量

        for table in ans['table']:
            for column in ans['column']:
                schema = table + '.' + column
                list_db = [item.lower() for item in db_schema_copy[example['db_id']]]
                if schema.lower() in list_db:
                    ground_truth.append(schema)
        
        stats.append(len(ground_truth))
        ground_truths.append(ground_truth)

    ### 处理预测结果
    with open(json_file, 'r', encoding='utf-8') as f:
        clms = json.load(f)

    # 按 question_id 进行排序
    clms = sorted(clms, key=lambda x: x["question_id"])

    pred_truths = []
    extracted_results = []

    for i, clm in enumerate(clms):
        pred_truth = clm['columns']  # 获取预测的列
        stats.append(len(pred_truth))
        pred_truths.append(pred_truth)

        # 存储 question_id、ground_truths 和 预测的 columns
        extracted_results.append({
            "question_id": clm["question_id"],
            "ground_truths": ground_truths[i],  # 真实值
            "columns": pred_truth  # 预测值
        })

    # 存储到 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f_out:
        json.dump(extracted_results, f_out, ensure_ascii=False, indent=4)

    # 计算评估指标
    num = 0
    num_table = 0
    num_column = 0
    num_all = 0
    num_nsr = 0

    for ground_truth, pred_truth in zip(ground_truths, pred_truths):
        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1

    # 打印各项评价指标
    print("SRR: ", num / len(ground_truths))         # Schema Recall Rate
    print("Avg.T: ", num_table / len(ground_truths)) # 平均表数
    print("Avg.C: ", num_column / len(ground_truths))# 平均列数
    print("NSR: ", num_nsr / num_all)                # Normalized Schema Recall

    print(f"抽取的列与真实值已存储到 {output_json_file}")

def recall_get_table1(json_file, output_json_file='extracted_columns.json'):

    """
        计算 SQL 语句的 Schema 预测结果，并评估预测效果
        - 存储 `ground_truths` (真实值)
        - 存储 `columns` (预测值)
        - 计算评估指标（SRR, Avg.T, Avg.C, NSR）
    """
    # 存储每个 gold sql 中涉及的所有 table_name.column_name
    stats = []
    stats_1 = []

    with open(DEV.dev_json_path, 'r', encoding='utf-8') as f:
        dev_set = json.load(f)

    # 提取 ground_truths
    ground_truths = []
    for example in dev_set:
        ground_truth = []
        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        stats_1.append(len(ans['table']))  # 记录 SQL 语句涉及的表数量

        for table in ans['table']:
            ground_truth.append(table)
        
        stats.append(len(ground_truth))
        ground_truths.append(ground_truth)

    ### 处理预测结果
    with open(json_file, 'r', encoding='utf-8') as f:
        clms = json.load(f)

    # 按 question_id 进行排序
    clms = sorted(clms, key=lambda x: x["question_id"])

    pred_truths = []
    extracted_results = []

    for i, clm in enumerate(clms):
        pred_truth = clm['tables']  # 获取预测的列
        stats.append(len(pred_truth))
        pred_truths.append(pred_truth)

        # 存储 question_id、ground_truths 和 预测的 columns
        extracted_results.append({
            "question_id": clm["question_id"],
            "ground_truths": ground_truths[i],  # 真实值
            "columns": pred_truth  # 预测值
        })

    # 存储到 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f_out:
        json.dump(extracted_results, f_out, ensure_ascii=False, indent=4)

    # 计算评估指标
    num = 0
    num_table = 0
    num_column = 0
    num_all = 0
    num_nsr = 0

    for ground_truth, pred_truth in zip(ground_truths, pred_truths):
        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1

    # 打印各项评价指标
    print("SRR: ", num / len(ground_truths))         # Schema Recall Rate
    print("Avg.T: ", num_table / len(ground_truths)) # 平均表数
    print("Avg.C: ", num_column / len(ground_truths))# 平均列数
    print("NSR: ", num_nsr / num_all)                # Normalized Schema Recall

    print(f"抽取的列与真实值已存储到 {output_json_file}")

def recall_get_table(json_file, output_json_file='extracted_columns.json'):
    """
        计算 SQL 语句的 Schema 预测结果，并评估预测效果
        - 存储 `ground_truths` (真实值)
        - 存储 `columns` (预测值)
        - 计算评估指标（SRR, Avg.T, Avg.C, NSR）
    """
    # 存储每个 gold sql 中涉及的所有 table_name.column_name
    stats = []
    stats_1 = []

    with open(DEV.dev_json_path, 'r', encoding='utf-8') as f:
        dev_set = json.load(f)

    # 提取 ground_truths
    ground_truths = []
    for example in dev_set:
        ground_truth = []
        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        stats_1.append(len(ans['table']))  # 记录 SQL 语句涉及的表数量

        for table in ans['table']:
            ground_truth.append(table)
        
        stats.append(len(ground_truth))
        ground_truths.append(ground_truth)

    ### 处理预测结果
    with open(json_file, 'r', encoding='utf-8') as f:
        clms = json.load(f)

    # 按 question_id 进行排序
    clms = sorted(clms, key=lambda x: x["question_id"])

    pred_truths = []

    for i, clm in enumerate(clms):
        # pred_truth = clm['columns']  # 获取预测的列
        # stats.append(len(pred_truth))
        table1 = clm['tables']
        table2 = set(item.split('.')[0] for item in clm['columns'])
        merged_tables = list(set(table1) | table2)
        pred_truths.append(merged_tables)


    # 计算评估指标
    num = 0
    num_table = 0
    num_column = 0
    num_all = 0
    num_nsr = 0

    results = []
    for ground_truth, pred_truth in zip(ground_truths, pred_truths):
        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        # num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1
        
         # 存储 question_id、ground_truths 和 预测的 columns
        results.append({
            
            "ground_truths": list(x1),  # 真实值
            "columns": list(x2),  # 预测值
            "match": x1.issubset(x2)
        })

    # 打印各项评价指标
    print("SRR: ", num / len(ground_truths))         # Schema Recall Rate
    print("Avg.T: ", num_table / len(ground_truths)) # 平均表数
    # print("Avg.C: ", num_column / len(ground_truths))# 平均列数
    print("NSR: ", num_nsr / num_all)                # Normalized Schema Recall

    print(f"抽取的列与真实值已存储到 {output_json_file}")

    

        # 存储到 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)


# 运行函数，并将结果保存到 extracted_columns.json
recall_get_column(json_file='src/dataset/sl_out_milvus_new2.json', output_json_file='src/dataset/extracted_columns_new_table.json')
# recall_get_table1(json_file='src/dataset/qwen/coder-32b/sl_out_milvus_new2.json', output_json_file='src/dataset/qwen/coder-7b/extracted_columns_milvus.json')