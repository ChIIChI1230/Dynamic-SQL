import pandas as pd
import sqlite3
import threading
import os
import sqlglot
import time
from config import DEV

def execute_sql_threaded(sql, db_name, result_container):
    db_path = f'{DEV.dev_databases_path}/{db_name}/{db_name}.sqlite'
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(sql)
        results = cursor.fetchall()
        result_container['row_count'] = len(results)
        result_container['column_count'] = len(results[0]) if results else 0
        result_container['result_preview'] = str(results[:5])

    except Exception as e:
        result_container['error'] = str(e)

    finally:
        if conn:
            conn.close()

def execute_sql(sql, db_name, timeout = 60):
    result_container = {}
    thread = threading.Thread(target = execute_sql_threaded, args=(sql, db_name, result_container))

    start_time = time.time()
    thread.start()
    thread.join(timeout)
    end_time = time.time()
    exec_time = end_time - start_time

    if thread.is_alive():
        # 超时处理
        return 0, 0, "TimeoutError: The SQL query took too long to execute. Please optimize your SQL query.", exec_time
    else:
        # 返回结果
        if 'error' in result_container:
            return 0, 0, "Error:" + result_container['error'], exec_time
        return result_container.get('row_count', 0), result_container.get('column_count', 0), result_container.get('result_preview', ""), exec_time


def simple_throw_row_data(db_name,tables,table_list):

    # 动态加载前三行数据
    simplified_ddl_data = []
    # 读取数据库
    mydb = sqlite3.connect(
        fr"{DEV.dev_databases_path}/{db_name}/{db_name}.sqlite")  # 链接数据库
    cur = mydb.cursor()
    # 表

    Tables = tables  # Tables 为元组列表
    for table in Tables:
        # 列
        col_name_list = table_list[table]
        column_str = ",".join(col_name_list)
        cur.execute(f"select {column_str} from `{table}`")
        # col_name_list = [tuple[0] for tuple in cur.description]
        # print(col_name_list)
        db_data_all = []
        # 获取前三行数据
        for i in range(3):
            db_data_all.append(cur.fetchone())
        # ddls_data
        test = ""
        for idx, column_data in enumerate(col_name_list):
            # print(list(db_data_all[2])[idx])
            try:
                test += f"{column_data}[{list(db_data_all[0])[idx]},{list(db_data_all[1])[idx]},{list(db_data_all[2])[idx]}],"
            except:
                test = test
        simplified_ddl_data.append(f"{table}({test[:-1]})")
    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"

    return ddls_data


def get_describe(db,tables,table_list):

    describe = "#\n# "
    for table in tables:
        with open(f'/Users/mac/Desktop/bird/dev_20240627/dev_databases/{db}/database_description/{table}.csv', 'r') as file:
            data = pd.read_csv(file)
        describe+= table + "("
        for column in table_list[table]:

            for index, row in data.iterrows():
                original_column_name= row['original_column_name']
                if column.replace('`','').lower().strip() == row['original_column_name'].lower().strip():
                    data_format = row['data_format']
                    if str(row['column_description']).strip() == 'nan':
                        describe+= f"`this data_format of the column is '{data_format}',the description of the column is '{column.replace('`','')}'`,"
                    else:
                        # print(str(row['column_description']))
                        describe+= f"`this data_format of the column is '{data_format}',the description of the column is '{str(row['column_description'])}'`,"
        describe = describe[:-1]+")\n# "
    return describe.strip()


# 从一个sqlite数据库文件中，提取出所有的表名和列名
def get_tables_and_columns(sqlite_db_path):
    with sqlite3.connect(sqlite_db_path) as conn:
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return [
            f"{_table[0]}.{_column[1]}"
            for _table in tables
            for _column in cursor.execute(f"PRAGMA table_info('{_table[0]}');").fetchall()
        ]


# 利用sqlglot工具，从一个sql语句中，提取出其中涉及的所有表和列（但是无法确认表与列之间的严格对应关系）
def extract_tables_and_columns(sql_query):
    parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
    table_names = parsed_query.find_all(sqlglot.exp.Table)
    column_names = parsed_query.find_all(sqlglot.exp.Column)
    return {
        'table': {_table.name for _table in table_names},
        'column': {_column.alias_or_name for _column in column_names}
    }


def get_all_schema():
    # 读取所有数据库
    db_base_path = DEV.dev_databases_path
    db_schema = {}
    for db_name in os.listdir(db_base_path):
        db_path = os.path.join(db_base_path, db_name, db_name + '.sqlite')
        if os.path.exists(db_path):
            db_schema[db_name] = get_tables_and_columns(db_path)
    return db_schema

def get_db_schema(db_name):
    db_base_path = DEV.dev_databases_path
    db_path = os.path.join(db_base_path, db_name, db_name + '.sqlite')

    return get_tables_and_columns(db_path)