import os
import sys
import json
import copy
from collections import defaultdict

# 获取项目根目录并添加到 sys.path
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from utils import get_all_schema, extract_tables_and_columns
from config import DEV


def recall_get_table(json_file1, json_file2):
    """
    计算 SQL 语句的 Schema 预测结果，匹配 question_id 进行评估，并进行去重处理。
    """
    db_schema_copy = copy.deepcopy(get_all_schema())

    with open(DEV.dev_json_path, 'r', encoding="utf-8") as f:
        dev_set = json.load(f)

    # 读取 Ground Truths (真实值)
    ground_truths = {}
    for example in dev_set:
        question_id = example['question_id']
        ground_truth = set()  # 使用 set 进行去重

        ans = extract_tables_and_columns(example['SQL'])  # 提取 SQL 涉及的表和列
        for table in ans['table']:
            for column in ans['column']:
                schema = f"{table}.{column}"
                list_db = {item.lower() for item in db_schema_copy[example['db_id']]}
                if schema.lower() in list_db:
                    ground_truth.add(schema)

        ground_truths[question_id] = ground_truth  # 存入字典

    # 读取 Pred Truths (预测值)
    pred_truth_map = defaultdict(set)  # 以 question_id 为 key，使用 set 去重

    """加载预测数据，并去重合并到 pred_truth_map"""
    with open(json_file1, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clm = json.loads(line)
                question_id = clm["question_id"]
                db_name = clm["db"]

                # pred_truths = {column.replace('`', '').lower()
                #                 for column in clm["columns"]
                #                     if column.replace('`', '').lower() in {item.lower() for item in db_schema_copy[db_name]}}
                
                pred_truths = [column.replace('`', '') for column in clm["llm_columns"]]

                pred_truth_map[question_id].update(pred_truths)  # 合并去重

    # # 处理 JSON 预测文件2
    # with open(json_file2, 'r', encoding='utf-8') as f:
    #     clms = json.load(f)
    
    # for clm in clms:
    #     question_id = clm["question_id"]
    #     pred_truth_map[question_id].update(set(clm["columns"]))  # ✅ 解决 AttributeError 问题

    # 计算评估指标
    num, num_table, num_column, num_all, num_nsr = 0, 0, 0, 0, 0
    t = []

    results = []
    for question_id, ground_truth in ground_truths.items():
        pred_truth = pred_truth_map.get(question_id, set())  # 通过 question_id 获取 pred_truth

        x1 = {item.lower() for item in ground_truth}
        x2 = {item.lower() for item in pred_truth}

        table = {item.split('.')[0] for item in pred_truth}
        num_table += len(table)
        num_column += len(x2)
        num_all += len(x1)
        num_nsr += len(x1.intersection(x2))

        if x1.issubset(x2):
            num += 1
            t.append(1)
        else:
            t.append(0)
        result = {
            "question_id": question_id,
            "ground_truth": list(ground_truth),
            "pred_truth": list(pred_truth),
            "match": x1.issubset(x2)
        }
        results.append(result)

        
    print(len(pred_truth_map))
    print("SRR: ", num / len(pred_truth_map))
    print("Avg.T: ", num_table / len(pred_truth_map))
    print("Avg.C: ", num_column / len(pred_truth_map))
    print("NSR: ", num_nsr / num_all)

    output_json_file = 'src/dataset/sl_out_milvus_evaluation.json'

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# 运行函数，确保 question_id 匹配
recall_get_table(
    # json_file1 = r'D:\code\vs_code\synthlink-sql\src\dataset\qwen\coder-7b\1_5_normalize_schema.jsonl',
    # json_file1 = r'D:\code\vs_code\synthlink-sql\src\dataset\deepseek\v3\1_sl_final_coder.jsonl',
    json_file1 = r'D:\code\vs_code\synthlink-sql\src\dataset\qwen\coder-7b\1_sl_final_coder.jsonl',
    # json_file1 = r'D:\code\vs_code\synthlink-sql\src\dataset\qwen\sl_out_milvus_new2.jsonl',
    json_file2 = 'src/dataset/sl_out_milvus.json'
)
