import os
import json
import sqlite3
from tqdm import tqdm

'''
稀疏模块用于将数据库中的数据导入milvus中，这些数据主要包括：
1. 枚举字段：
2. 布尔字段：
3. 分类字段：
4. 数值字段：
5. 长文本字段：
'''
# 构造基础目录和输出目录（跨平台拼接路径）
base_dir = os.path.join(os.getcwd(), 'database', 'dev_databases')
output_base = os.path.join(os.getcwd(), 'milvus', 'sparse')
count_dir = os.path.join(output_base, 'count')

def count_data_from_db():
    # 1. 创建输出目录
    try:
        os.makedirs(output_base, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] 创建输出目录失败: {e}")
        return

    # 2. 检查 base_dir
    if not os.path.isdir(base_dir):
        print(f"[ERROR] 数据库根目录不存在: {base_dir}")
        return

    # 3. 遍历每个子目录（每个数据库）
    entries = [e for e in os.listdir(base_dir) if not e.startswith('.')]
    for entry in tqdm(entries, desc="Databases", unit="db"):
        db_path = os.path.join(base_dir, entry, f"{entry}.sqlite")
        if not os.path.isfile(db_path):
            print(f"[WARN] 找不到数据库文件: {db_path}")
            continue

        # 4. 打开数据库
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        except Exception as e:
            print(f"[{entry}] 打开数据库失败: {e}")
            continue

        # 5. 获取所有表名
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[{entry}] 获取表列表失败: {e}")
            conn.close()
            continue

        result = {}

        # 6. 遍历每张表并统计列信息
        for table in tqdm(tables, desc=f"{entry} → Tables", unit="tbl", leave=False):
            result[table] = {}

            # 获取列信息
            try:
                cursor.execute(f"PRAGMA table_info({table});")
                cols = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
            except Exception as e:
                print(f"[{entry}.{table}] PRAGMA table_info 失败: {e}")
                continue

            # 遍历每列
            for cid, col_name, col_type, *_ in cols:
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT `{col_name}`) FROM {table};")
                    distinct_count = cursor.fetchone()[0]
                except Exception as e:
                    distinct_count = f"ERROR: {e}"
                result[table][col_name] = {
                    "type": col_type,
                    "distinct_count": distinct_count
                }

        conn.close()

        # 7. 写入 JSON 文件
        out_file = os.path.join(output_base, f"{entry}.json")
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"[{entry}] 已生成文件: {out_file}")
        except Exception as e:
            print(f"[{entry}] 写入 JSON 失败: {e}")

def select_data_from_db():
    # 1. 创建输出目录
    try:
        os.makedirs(output_base, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] 创建输出目录失败: {e}")
        return

    # 2. 检查 count_dir
    if not os.path.isdir(count_dir):
        print(f"[ERROR] 列计数目录不存在: {count_dir}")
        return

    # 3. 遍历每个子目录（每个 count JSON 文件）
    entries = [e for e in os.listdir(count_dir) if not e.startswith('.')]
    for entry in tqdm(entries, desc="Processing DBs", unit="file"):
        # 从文件名中提取数据库名（假设文件名格式为 <db_name>.json）
        db_name, ext = os.path.splitext(entry)
        if ext.lower() != '.json':
            print(f"[WARN] 跳过非 JSON 文件: {entry}")
            continue

        # 构造对应的 SQLite 路径
        db_path = os.path.join(base_dir, db_name, f"{db_name}.sqlite")
        if not os.path.isfile(db_path):
            print(f"[WARN] 找不到数据库文件: {db_path}")
            continue

        # 4. 打开数据库
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        except Exception as e:
            print(f"[{db_name}] 打开数据库失败: {e}")
            continue

        # 5. 读取 count 文件
        try:
            with open(os.path.join(count_dir, entry), 'r', encoding='utf-8') as f:
                count_data = json.load(f)
        except Exception as e:
            print(f"[{db_name}] 读取列计数文件失败: {e}")
            conn.close()
            continue

        results = []

        # 6. 遍历每张表并补充 DISTINCT 值
        for table, cols in tqdm(count_data.items(),
                                 desc=f"{db_name} → Tables",
                                 unit="tbl",
                                 leave=False):
            # 构造引用表名（防止 SQL 关键字冲突）
            q_table = f'"{table}"'
            for column_name, meta in cols.items():
                # 获取已有的值列表，保证是列表类型，否则置空
                existing_values = meta.get('values')
            
                # 直接拷贝以免修改原始数据
                values = list(existing_values)
                result = {
                    'table_name': table,
                    'column_name': column_name,
                    'values': values
                }

                # 仅当已有 values 为空时再查询数据库
                if not result['values']:
                    # 添加格式化后的列名分割结果，不保留中括号
                    split_result = " ".join(column_name.split('_'))
                    if split_result:
                        result['values'].append(split_result)
                    q_col = f'"{column_name}"'
                    try:
                        # 查询该列中不同值的数量，并按出现频率降序排列，限制返回 1000 条记录
                        query = (
                            f"SELECT {q_col}, COUNT(*) as frequency FROM {q_table} "
                            f"GROUP BY {q_col} "
                            f"ORDER BY frequency DESC LIMIT 1000;"
                        )
                        cursor.execute(query)
                        fetched = cursor.fetchall()
                        # 获取 DISTINCT 值（排除 None 值）
                        distinct_values = [row[0] for row in fetched if row[0] is not None]
                    except Exception as e:
                        print(f"[{db_name}.{table}.{column_name}] 查询 DISTINCT 失败: {e}")
                        distinct_values = []
                    result['values'].extend(distinct_values)

                # 如果结果 non-empty，则保存到列表中
                if result['values']:
                    results.append(result)
        conn.close()

        # 7. 写入 JSON 文件（按数据库分文件）
        # out_file = os.path.join(output_base, 'values', f"{db_name}.json")
        out_file = os.path.join(output_base, f"{db_name}.json")
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"[{db_name}] 已生成文件: {out_file}")
        except Exception as e:
            print(f"[{db_name}] 写入 JSON 失败: {e}")


if __name__ == "__main__":
    
    select_data_from_db()

