import os
import sys
import json
import copy

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from utils import get_all_schema, extract_tables_and_columns
from config import DEV


def recall_get_column(json_file):
    """
    计算 SQL 语句的 Schema 预测结果，匹配 question_id 进行评估
    """
    db_schema_copy = copy.deepcopy(get_all_schema())

    with open(DEV.dev_json_path, 'r', encoding="utf-8") as f:
        dev_set = json.load(f)

    # 读取 Ground Truths (真实值)
    ground_truths = {}
    for example in dev_set:
        question_id = example['question_id']
        ground_truth = []

        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        for table in ans['table']:
            for column in ans['column']:
                schema = f"{table}.{column}"
                list_db = [item.lower() for item in db_schema_copy[example['db_id']]]
                if schema.lower() in list_db:
                    ground_truth.append(schema)

        ground_truths[question_id] = ground_truth

    # 读取 Pred Truths (预测值)
    pred_truth_map = {}  # 以 question_id 为 key
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clm = json.loads(line)
                question_id = clm["question_id"]
                pred_truth = []
                db_name = clm["db"]

                for column in clm["columns"]:
                    schema = column.replace('`', '')
                    if schema.lower() in [item.lower() for item in db_schema_copy[db_name]]:
                        pred_truth.append(schema)

                pred_truth_map[question_id] = pred_truth

    # 计算评估指标
    num, num_table, num_column, num_all, num_nsr = 0, 0, 0, 0, 0
    t = []

    for question_id, ground_truth in ground_truths.items():
        pred_truth = pred_truth_map.get(question_id, [])  # 通过 question_id 获取 pred_truth

        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1
            t.append(1)
        else:
            t.append(0)

    print("SRR: ", num / len(ground_truths))
    print("Avg.T: ", num_table / len(ground_truths))
    print("Avg.C: ", num_column / len(ground_truths))
    print("NSR: ", num_nsr / num_all)

def recall_get_table(json_file):
    """
    计算 SQL 语句的 Schema 预测结果，匹配 question_id 进行评估
    """
    db_schema_copy = copy.deepcopy(get_all_schema())

    with open(DEV.dev_json_path, 'r', encoding="utf-8") as f:
        dev_set = json.load(f)

    # 读取 Ground Truths (真实值)
    ground_truths = {}
    for example in dev_set:
        question_id = example['question_id']
        ground_truth = []

        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        for table in ans['table']:
            ground_truth.append(table)

        ground_truths[question_id] = ground_truth

    # 读取 Pred Truths (预测值)
    pred_truth_map = {}  # 以 question_id 为 key
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clm = json.loads(line)
                question_id = clm["question_id"]
                pred_truth = []
                db_name = clm["db"]

                # for column in clm["columns"]:
                #     schema = column.replace('`', '')
                #     if schema.lower() in [item.lower() for item in db_schema_copy[db_name]]:
                #         pred_truth.append(schema)

                for table in clm["tables"]:
                    pred_truth.append(table)

                pred_truth_map[question_id] = pred_truth

    # 计算评估指标
    num, num_table, num_column, num_all, num_nsr = 0, 0, 0, 0, 0
    t = []

    for question_id, ground_truth in ground_truths.items():
        pred_truth = pred_truth_map.get(question_id, [])  # 通过 question_id 获取 pred_truth

        x1 = set(item.lower() for item in ground_truth)
        x2 = set(item.lower() for item in pred_truth)

        table = set(item.split('.')[0] for item in pred_truth)
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1
            t.append(1)
        else:
            t.append(0)

    print("SRR: ", num / len(ground_truths))
    print("Avg.T: ", num_table / len(ground_truths))
    print("Avg.C: ", num_column / len(ground_truths))
    print("NSR: ", num_nsr / num_all)


# 运行函数，确保 question_id 匹配
recall_get_table(json_file=r'D:\code\vs_code\synthlink-sql\src\dataset\deepseek\result\enhance_query_link_schema_result.jsonl')
