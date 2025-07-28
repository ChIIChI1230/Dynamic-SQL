import sqlite3
from config import DEV

# 连接到指定数据库，返回数据库连接对象
def connect_to_db(db_name):

    # 构建数据库文件的路径，并建立数据库连接
    return sqlite3.connect(DEV.dev_databases_path+'/' + db_name + f'/{db_name}.sqlite')

# 获取数据库中所有表的名称
def get_all_table_names(db_name):
    conn = connect_to_db(db_name)
    cursor = conn.cursor()

    # 查询 sqlite_master 表，获取所有类型为 'table' 且名称不为 'sqlite_sequence' 的表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    table_names = cursor.fetchall()

    conn.close()

    return [name[0] for name in table_names]

# 获取指定表的所有列的名称
def get_all_column_names(db_name, table_name):
    conn = connect_to_db(db_name)
    cursor = conn.cursor()

    # 使用 PRAGMA table_info 获取指定表的结构信息
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    table_info = cursor.fetchall()

    # 从查询结果中提取每个列的名称（每个元组的第二个元素为列名）
    column_names = [column[1] for column in table_info]

    conn.close()

    return column_names

# 获取指定表的外键信息
def get_foreign_key_info(db_name, table_name):
    conn = connect_to_db(db_name)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
    foreign_key_info = cursor.fetchall()

    conn.close()

    return foreign_key_info

# 获取指定表的所有列的名称，并生成一个简化的 DDL 字符串
def get_table_infos(database_name):
    table_list = get_all_table_names(database_name)
    table_str = '#\n# '
    for table in table_list:
        column_list = get_all_column_names(database_name, table)

        column_list = ['`' + column + '`' for column in column_list]

        columns_str = f'{table}(' + ', '.join(column_list) + ')'

        table_str += columns_str + '\n# '

    return table_str

# 获取数据库中所有外键信息的字符串描述
def get_foreign_key_infos(database_name):
    table_list = get_all_table_names(database_name)

    foreign_str = '#\n# '
    for table in table_list:
        foreign_lists = get_foreign_key_info(database_name, table)

        for foreign in foreign_lists:
            foreign_one = f'{table}({foreign[3]}) references {foreign[2]}({foreign[4]})'
            foreign_str += foreign_one + '\n# '

    return foreign_str

# 获取指定数据库中每个表的前三行数据，并生成一个简化的数据描述字符串
def get_throw_row_data(db_name):
    # 动态加载前三行数据
    simplified_ddl_data = []

    # 连接数据库
    mydb = connect_to_db(db_name) 
    cur = mydb.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    Tables = cur.fetchall()  # Tables 为元组列表
    
    for table in Tables:
        # 列
        cur.execute(f"select * from `{table[0]}`")
        col_name_list = [tuple[0] for tuple in cur.description]
        # print(col_name_list)
        db_data_all = []
        # 获取前三行数据
        for i in range(3):
            db_data_all.append(cur.fetchone())
        # ddls_data
        test = ""
        for idx, column_data in enumerate(col_name_list):
            try:
                test += f"`{column_data}`[{list(db_data_all[0])[idx]},{list(db_data_all[1])[idx]},{list(db_data_all[2])[idx]}],"
            except:
                test = test
        simplified_ddl_data.append(f"{table[0]}({test[:-1]})")
    ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n"

    return ddls_data

def get_five_row_data(db_name, columns):
    """
    获取指定数据库中指定表、指定列的随机五条数据，并生成一个简化的数据描述字符串。
    如果表或列不存在，会跳过该列并打印警告。
    """
    mydb = connect_to_db(db_name)
    cur = mydb.cursor()
    
    # 缓存每张表的列列表，避免重复查询
    table_columns_cache = {}
    results = {}

    for col in columns:
        table, column = col.split(".")
        results.setdefault(table, {})

        # 1. 确保我们已经拿到该表的列名列表
        if table not in table_columns_cache:
            try:
                # SQLite: PRAGMA table_info；MySQL 可用 DESCRIBE
                cur.execute(f"PRAGMA table_info(`{table}`)")
                info = cur.fetchall()
                # info 列表里每行第 2 列是列名
                table_columns_cache[table] = {row[1] for row in info}
            except Exception as e:
                print(f"Warning: 无法获取表结构 {table}: {e}")
                table_columns_cache[table] = set()

        # 2. 如果列不在该表中，跳过
        if column not in table_columns_cache[table]:
            print(f"Warning: 表 `{table}` 中不存在列 `{column}`，已跳过。")
            continue

        # 3. 随机抽样
        try:
            cur.execute(
                f"SELECT DISTINCT `{column}` "
                f"FROM `{table}` "
                f"ORDER BY RANDOM() "   # SQLite 用 RANDOM()；MySQL 请改为 RAND()
                f"LIMIT 5"
            )
            rows = cur.fetchall()
        except Exception as e:
            print(f"Warning: 随机抽样失败，降级到 LIMIT 查询 for {table}.{column}: {e}")
            try:
                cur.execute(f"SELECT `{column}` FROM `{table}` LIMIT 5")
                rows = cur.fetchall()
            except Exception as e2:
                print(f"Error: 连降级查询也失败 for {table}.{column}: {e2}")
                rows = []

        # 4. 拼接结果
        data_values = [str(row[0]) for row in rows]
        col_str = f"`{column}`[{','.join(data_values)}]"
        results[table][column] = col_str

    # 5. 构造最终的描述字符串
    ddls_data = "#"
    for table, cols in results.items():
        if not cols:
            continue
        ddls_data += f"\n# {table}({','.join(cols.values())});"

    return ddls_data + "\n"