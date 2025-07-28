from utils.util import simple_throw_row_data
import json
from utils.db_op import get_all_column_names
from difflib import get_close_matches

# 从输入数据中提取数据库名、外键信息、表名列表和列名列表
def simplified(ppl):
    db = ppl['db']
    foreign_key = ppl['foreign_key'].strip()
    tables = ppl['tables']
    columns = ppl['columns']

    # 简化ddl
    table_list = {}
    simple_ddl_simple = "#\n# "
    for table in tables:
        simple_ddl_simple += table + "("
        column_list = []
        for column in columns:
            _table = column.split(".")[0].strip()
            if _table == table:
                column = column.split(".")[1].strip()
                column_list.append(column)
                simple_ddl_simple += column + ","
        table_list[table] = column_list
        simple_ddl_simple = simple_ddl_simple[:-1] + ")\n# "
    simple_ddl = simple_ddl_simple.strip()

    # 简化data
    # data_ddl = simple_throw_row_data(db, tables, table_list)
    # ddl_data = "#\n" + data_ddl.strip() + "\n# "

    # 简化foreign_key
    temp = "#\n"
    for line in foreign_key.split("\n"):
        try:
            table1 = line.split("# ")[1].split("(")[0].strip()
            table2 = line.split("references ")[1].split("(")[0].strip()
            if table1.lower() in tables and table2.lower() in tables:
                temp += line + "\n"
        except:
            continue
    foreign_key = temp.strip() + "\n# "
    # return simple_ddl, ddl_data, foreign_key
    return simple_ddl, foreign_key

def simplified_ddl1(db, tables, columns, foreign_key):

    simple_ddl_simple = "#\n# "
    for table in tables:
        simple_ddl_simple += table + "("
        column_list = get_all_column_names(db, table)
        for column in column_list:
            simple_ddl_simple += column + ","
        simple_ddl_simple = simple_ddl_simple[:-1] + ")\n# "
    simple_ddl = simple_ddl_simple.strip()
    
    # 简化foreign_key
    # 简化foreign_key
    temp = "#\n"
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
            if table1.lower() in tables and table2.lower() in tables:
                if f"{table1}.{column1}" not in columns:
                    columns.append(f"{table1}.{column1}")
                if f"{table2}.{column2}" not in columns:
                    columns.append(f"{table2}.{column2}")
                temp += line + "\n"
        except:
            continue
    foreign_key = temp.strip() + "\n# "


    with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
        column_meaning = json.load(f)

    ### 收集解释
    explanation = ""
    for table_name in tables:
        column_list = get_all_column_names(db, table_name)
        for column_name in column_list:
            h = f"{db}|{table_name}|{column_name}"
            if h in column_meaning:
                explanation += f"### {table_name}.{column_name}: {column_meaning[h]}\n"

    explanation = explanation.replace("### ", "# ")


    return simple_ddl, foreign_key, explanation, columns
    
def simplified_ddl2(db, tables, columns, foreign_key):
    # 简化foreign_key
    temp = "#\n"
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
            if table1.lower() in tables and table2.lower() in tables:
                if f"{table1}.{column1}" not in columns:
                    columns.append(f"{table1}.{column1}")
                if f"{table2}.{column2}" not in columns:
                    columns.append(f"{table2}.{column2}")
                temp += line + "\n"
        except:
            continue
    foreign_key = temp.strip() + "\n# "

    simple_ddl_simple = "#\n# "
    for table in tables:
        simple_ddl_simple += table + "("
        # 找出属于当前表的所有列，并提取列名（即点号后面的部分）
        cols = [col.split('.')[1] for col in columns if col.split('.')[0] == table]
        simple_ddl_simple += ", ".join(cols)
        simple_ddl_simple += ")\n# "
    simple_ddl = simple_ddl_simple.strip()
    
    with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
        column_meaning = json.load(f)

    ### 收集解释
    explanation = ""
    for col in columns:
            h = f"{db}|{col.split('.')[0]}|{col.split('.')[1]}"
            if h in column_meaning:
                explanation += f"### {col}: {column_meaning[h]}\n"

    explanation = explanation.replace("### ", "# ")

    return simple_ddl, foreign_key, explanation, columns
def explanation_collection(ppl):

    with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
        column_meaning = json.load(f)

    tables = ppl['tables']
    columns = ppl['columns']
    db = ppl['db']

    columns = [obj.replace('`', '').lower() for obj in columns]
    columns = [obj.replace('.', '.`') + '`' for obj in columns]

    ### 收集解释
    explanation = ""

    for x in column_meaning:
        h = x
        x = x.lower().split("|")
        db_name = x[0]
        table_name = x[1]
        column_name = x[2]
        if db == db_name:
            if table_name in tables:
                if table_name + '.`' + column_name + '`' in columns:
                    explanation += f"### {table_name}.{column_name}: {column_meaning[h]}\n"

    explanation = explanation.replace("### ", "# ")

    return explanation

def explanation_collection_all(ppl):

    with open('data/column_meaning.json', 'r', encoding = "utf-8") as f:
        column_meaning = json.load(f)

    tables = ppl['tables']
    columns = ppl['columns']
    db = ppl['db']

    ### 收集解释
    explanation = ""

    for x in column_meaning:
        h = x
        x = x.lower().split("|")
        db_name = x[0]
        table_name = x[1]
        # column_name = x[2]
        if db == db_name:
            if table_name in tables:
                columns = get_all_column_names(db, table_name)
                columns = [obj.replace('`', '').lower() for obj in columns]
                columns = [obj.replace('.', '.`') + '`' for obj in columns]
                for column_name in columns:
                    if table_name + '.`' + column_name + '`' in columns:
                        explanation += f"### {table_name}.{column_name}: {column_meaning[h]}\n"

    explanation = explanation.replace("### ", "# ")

    return explanation


def correct_columns(db, tables, columns):

    with open('data/dev_columns.json', 'r', encoding = "utf-8") as f:
        all_columns = json.load(f)
    
    db_columns = all_columns[db]
    correct_tables = set(tables)
    correct_columns = set()
    correct_lookup = {col.lower(): col for col in db_columns}

    for col in columns:
        col_lower = col.lower()
        if col_lower in correct_lookup:
            correct_columns.add(correct_lookup[col_lower])
            correct_tables.add(correct_lookup[col_lower].split('.')[0])
        else:
            # col_part = col.split('.')[1]
            # matched_fields = [c for c in db_columns if c.split('.')[1] == col_part]
            # if matched_fields:
            #     print(col + " -> 多个匹配: " + ", ".join(matched_fields))
            #     for m in matched_fields:
            #         correct_columns.add(m)
            #         correct_tables.add(m.split('.')[0])
            # else:
            #     matches = get_close_matches(col_lower, list(correct_lookup.keys()), n=1, cutoff=0.7)
            #     if matches:
            #         print(col + "->" + correct_lookup[matches[0]])
            #         correct_columns.add(correct_lookup[matches[0]])
            #         correct_tables.add(correct_lookup[matches[0]].split('.')[0])
            #     else:
            #         correct_columns.add(col)
            #         correct_tables.add(col.split('.')[0])

            col_part = col.split('.')[1]
            matched_fields = [c for c in db_columns if c.split('.')[1] == col_part]
            matches = get_close_matches(col_lower, list(correct_lookup.keys()), n = 2, cutoff=0.7)
            matches_full = [correct_lookup[m] for m in matches]

            matched_fields_final = set(matched_fields) | set(matches_full)
            if matched_fields_final:
                for m in matched_fields_final:
                    correct_columns.add(m)
                    correct_tables.add(m.split('.')[0])
            else:
                correct_columns.add(col)
                correct_tables.add(col.split('.')[0])

    return list(correct_tables), list(correct_columns)