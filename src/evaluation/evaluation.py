import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm  # 导入 tqdm
import math

# 全局变量，存放多进程返回结果和进度条对象
exec_result = []
pbar = None  # 用于进度条更新

# ----------------------------
# 数据加载与解析相关函数
# ----------------------------

def load_json(file_path):
    """
    从指定路径加载 JSON 文件并解析为 Python 对象
    """
    with open(file_path, 'r', encoding='utf-8') as j:
        contents = json.loads(j.read())
    return contents

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data

def result_callback(result):
    """
    多进程回调函数，收集结果并更新进度条
    """
    global exec_result, pbar
    exec_result.append(result)
    if pbar is not None:
        pbar.update(1)

# ----------------------------
# SQL 执行相关函数
# ----------------------------

def execute_sql(predicted_sql, ground_truth, db_path):
    """
    在指定的 SQLite 数据库中执行预测 SQL 和真实 SQL，并比较结果
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
    finally:
        conn.close()
    return 1 if set(predicted_res) == set(ground_truth_res) else 0

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    """
    单个 SQL 查询对执行模型，带超时保护
    """
    try:
        res = func_timeout(meta_time_out, execute_sql, args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
    except Exception as e:
        res = 0
    result = {'sql_idx': idx, 'res': res}
    return result

# ----------------------------
# SQL 数据与数据库路径加载函数
# ----------------------------

def package_sqls(sql_path, db_root_path, mode, data_mode='dev'):
    clean_sqls = {}       # 存储 SQL 查询
    db_path_list = {}     # 存储数据库路径
    difficulty_dict = {}  # 存储难度分类

    if mode == 'gt':
        sql_data = load_json(sql_path)
        for item in sql_data:
            if isinstance(item['SQL'], str):
                question_id = item['question_id']
                sql = item['SQL']
                db_name = item['db_id']
                difficulty = item.get('difficulty', 'simple')
            else:
                question_id, sql, db_name, difficulty = 1534, " ", "financial", "simple"

            clean_sqls[question_id] = sql
            db_path_list[question_id] = db_root_path + db_name + '/' + db_name + '.sqlite'
            difficulty_dict[question_id] = difficulty
    else:
        sql_data = load_json_lines(sql_path)

        for item in sql_data:
            if isinstance(item.get("sql_1", item.get("sql_2", "")), str):
                question_id = item['question_id']
                sql = item.get("sql_1", item.get("sql_2", ""))
                db_name = item['db']
                difficulty = item.get('difficulty', 'simple')
            else:
                question_id, sql, db_name, difficulty = 1534, " ", "financial", "simple"

            clean_sqls[question_id] = sql
            db_path_list[question_id] = db_root_path + db_name + '/' + db_name + '.sqlite'
            difficulty_dict[question_id] = difficulty

    return clean_sqls, db_path_list, difficulty_dict

def match_predictions_with_ground_truth(predicted_sqls, ground_truth_sqls, predicted_db_paths, predicted_difficulties):
    matched_sqls = []
    matched_db_paths = []
    matched_difficulties = []

    for question_id, predicted_sql in predicted_sqls.items():
        if question_id in ground_truth_sqls:
            ground_truth_sql = ground_truth_sqls[question_id]
            predicted_db_path = predicted_db_paths.get(question_id, "")
            predicted_diff = predicted_difficulties.get(question_id, "simple")
            
            matched_sqls.append((predicted_sql, ground_truth_sql))
            matched_db_paths.append(predicted_db_path)
            matched_difficulties.append(predicted_diff)
    return matched_sqls, matched_db_paths, matched_difficulties

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    global pbar, exec_result
    pool = mp.Pool(processes=num_cpus)
    num_tasks = len(sqls)
    exec_result = []  # 清空之前的结果

    # 初始化 tqdm 进度条
    pbar = tqdm(total=num_tasks, desc="执行 SQL 查询", ncols=100)

    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

    # 任务结束后关闭进度条
    pbar.close()

def sort_results(list_of_dicts):
    return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

# ----------------------------
# 准确率计算与结果输出相关函数
# ----------------------------

def compute_acc_by_diff(exec_results, difficulty_list):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]

    with open('src/evaluation/chess.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    simple_results, moderate_results, challenging_results = [], [], []

    for i, diff in enumerate(difficulty_list):
        if diff == 'simple':
            simple_results.append(exec_results[i])
        elif diff == 'moderate':
            moderate_results.append(exec_results[i])
        elif diff == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results]) / len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results]) / len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results]) / len(challenging_results) if challenging_results else 0
    all_acc = sum(results) / num_queries if num_queries else 0
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

def compute_ves(exec_results):
    num_queries = len(exec_results)
    total_ratio = 0
    count = 0

    for i, result in enumerate(exec_results):
        if result['time_ratio'] != 0:
            count += 1
        total_ratio += math.sqrt(result['time_ratio']) * 100
    ves = (total_ratio/num_queries)
    return ves

def compute_ves_by_diff(exec_results, difficulty_list):
    num_queries = len(exec_results)
    # contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []
    for i, diff in enumerate(difficulty_list):
        if diff == 'simple':
            simple_results.append(exec_results[i])
        elif diff == 'moderate':
            moderate_results.append(exec_results[i])
        elif diff == 'challenging':
            challenging_results.append(exec_results[i])
    simple_ves = compute_ves(simple_results)
    moderate_ves = compute_ves(moderate_results)
    challenging_ves = compute_ves(challenging_results)
    all_ves = compute_ves(exec_results)
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_ves, moderate_ves, challenging_ves, all_ves, count_lists

def print_data(score_lists, count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

# ----------------------------
# 主程序入口
# ----------------------------

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument('--predicted_sql_path', type=str, required=False, default='src/dataset/qwen/coder-32b/en/3_cot_synthesize_sql1.jsonl')
    # args_parser.add_argument('--predicted_sql_path', type=str, required=False, default='src/dataset/qwen/coder-32b/en/2_sql_generation1.jsonl')
    args_parser.add_argument('--predicted_sql_path', type=str, required=False, default='src/dataset/gpt/1_sl_final_coder.jsonl')
    # args_parser.add_argument('--predicted_sql_path', type=str, required=False, default='src/dataset/qwen/coder-7b/baseline_sql.jsonl')
    # args_parser.add_argument('--predicted_sql_path', type=str, required=False, default='src/dataset/qwen/coder-32b/en/4_final_sql1.jsonl')
    args_parser.add_argument('--ground_truth_path', type=str, required=False, default='data/dev.json')
    args_parser.add_argument('--data_mode', type=str, required=False, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=False, default='database/dev_databases/')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=60.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--difficulty', type=str, default='simple')
    args_parser.add_argument('--diff_json_path', type=str, default='results/')
    args = args_parser.parse_args()

    # 加载预测 SQL 查询和数据库路径
    pred_queries, db_paths, difficulty = package_sqls(args.predicted_sql_path, args.db_root_path, mode=args.mode_predict, data_mode=args.data_mode)
    # 加载真实 SQL 查询和数据库路径
    gt_queries, db_paths_gt, difficulty_gt = package_sqls(args.ground_truth_path, args.db_root_path, mode=args.mode_gt, data_mode=args.data_mode)

    # 根据 question_id 匹配预测和真实数据
    matched_queries, matched_db_paths, difficulty_list = match_predictions_with_ground_truth(pred_queries, gt_queries, db_paths, difficulty)

    # 使用多进程并行执行 SQL 查询，带有进度条显示
    run_sqls_parallel(matched_queries, db_places=matched_db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)

    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(exec_result, difficulty_list)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    # simple_ves, moderate_ves, challenging_ves, ves, count_lists = compute_ves_by_diff(exec_result, difficulty_list)
    # score_lists = [simple_ves, moderate_ves, challenging_ves, ves]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
