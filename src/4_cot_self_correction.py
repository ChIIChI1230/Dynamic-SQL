import os
import re
import sys
import json
import argparse
import logging
from tqdm import tqdm
import concurrent.futures
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from llm import QWEN_LLM_CODER
from utils.util import execute_sql

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 判断 sql 语句是否需要修复的提示词
instruction_need = """
你是一名 SQL 查询语义分析专家，任务是判断一个已成功执行并返回结果的 SQL 查询语句是否准确满足用户的查询意图。

请基于以下输入信息进行分析：
1. 用户问题：自然语言表述的查询意图；
2. SQL 查询语句：已成功执行并返回结果的 SQL；
3. SQL 查询结果：数据表格形式，展示查询返回的若干行数据；
4. 数据库结构信息：表名、字段名、字段类型、 表间关系等；
5. 字段解释与枚举值：帮助理解字段含义与可选值；
6. 相关列样本数据：每列的示例值、空值、格式异常等；
7. 与问题相关的领域知识（如有）：对业务背景的补充说明。

请仔细分析 SQL 查询结果是否覆盖了用户意图中的关键信息、筛选条件、聚合逻辑等。如果查询结果与用户问题之间存在语义偏差、字段遗漏、错误聚合或逻辑误解，则应建议进行修复。

输出格式如下：
【是否修复】：需要修复 / 无需修复  
【理由】：（简要说明 SQL 查询结果是否存在语义偏差、字段遗漏、聚合逻辑错误等）
"""

# 修复路径1：直接修复 SQL_3 的提示词
instruction_fix = """
你是一名 SQL 修复专家，任务是根据用户的输入信息，对当前 SQL 查询进行修复，使其更准确地表达用户意图，同时确保其能够在给定的数据库结构上正确执行。

### 核心任务：
1. 判断当前 SQL 是否存在语义错误或语法错误；
2. 修复字段选择、过滤条件、表连接、聚合逻辑、语法结构等问题；
3. 若 SQL 无法执行（如语法错误、字段不存在、连接错误等），请修复这些结构性问题；
4. 生成新的 SQL 查询，其语义更贴近用户意图，且能够在数据库中成功执行;
5. 所有使用的字段名和表名必须严格来自于数据库结构信息，不允许使用未出现的字段或表。

### 输入：
- 用户问题：自然语言描述的查询意图；
- SQL 查询、执行结果及分析判断：用于判断其与查询意图的一致性；
- 数据库结构信息：表名、字段名、字段类型、 表间关系等；
- 字段解释与枚举值：帮助理解字段含义与可选值；
- 相关列样本数据：每列的示例值、空值、格式异常等；
- 与问题相关的领域知识（如有）：对业务背景的补充说明。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请仅以 JSON 格式返回最终融合优化后的 SQL 查询，格式如下：{"sql": "新生成的 SQL 查询语句"}

### 注意事项：
- 生成的 SQL 必须严格符合数据库结构信息，不得使用不存在的名称；
- 不要输出任何解释、注释或多余内容，仅输出 JSON 对象。
"""

# 修复路径2：基于 SQL_1 / SQL_2 重构的提示词
instruction_reconstruct = """
你是一位精通 SQL 语义理解与结构优化的专家，擅长根据用户查询意图，分析、对比并融合多条 SQL 查询，生成结构合理、语义精准、执行正确的最终查询语句。
当前生成的 SQL 查询 SQL_3（由 SQL_1 和 SQL_2 合成）以及 SQL_4（基于 SQL_3 的修复版本）均未满足用户的查询需求。请你回退参考 SQL_1 与 SQL_2，深入分析 SQL_3 和 SQL_4 中的不足，并在此基础上融合构造一个更准确、更健壮的 SQL 查询。

### 核心任务：
1. 分析 SQL_3 和 SQL_4 的失败原因：识别其在字段选择、过滤逻辑、连接方式、聚合与排序等方面的错误或遗漏；
2. 对比 SQL_1 和 SQL_2 的结构、逻辑与结果表现，评估它们对用户意图的覆盖程度；
3. 提取 SQL_1 与 SQL_2 中准确合理的部分，**吸取 SQL_3 和 SQL_4 的失败教训，避免重蹈错误**；
4. 融合优化查询逻辑与结构，构造语义准确、字段规范、执行无误的最终 SQL 查询；
5. 确保最终 SQL 可正确执行，结构清晰，语义完整，不包含冗余或无用部分。

### 输入：
- 用户问题：自然语言描述的查询意图；
- SQL_1 与 SQL_2：原始 SQL 查询及其执行结果；
- SQL_3 与 SQL_4：失败的合成版本与修复版本；
- 数据库结构信息：表名、字段名、字段类型、表间关系等；
- 字段解释与枚举值：说明字段含义与可选值；
- 相关列样本数据：示例值、是否为空、格式异常等；
- 与问题相关的业务知识（如有）：补充语义理解与背景约束。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请仅以 JSON 格式返回最终优化融合后的 SQL 查询；
- 输出格式如下：{"sql": "最终合成的 SQL 查询语句"}

### 操作步骤：
1. 分析 SQL_3 和 SQL_4 的问题点：
    - 是否语义错误、字段选择偏差、逻辑混乱、表连接有误；
    - 是否出现语法无法执行、字段缺失或空结果；
2. 分析 SQL_1 与 SQL_2：
    - 各自是否有正确的结构、查询逻辑、过滤条件或聚合思路；
    - 是否部分结果是正确的，能满足部分查询需求；
3. 综合对比四条 SQL 与用户意图的匹配度，提取有价值部分；
4. 在避免 SQL_3 和 SQL_4 错误的基础上，融合 SQL_1 和 SQL_2 的正确结构；
5. 输出一个语义完整、执行可行的最终 SQL 查询。

### 注意事项：
1. 所有使用的字段名和表名必须严格来自于数据库结构信息，不允许使用未出现的字段或表。
2. 仅在必要部分进行修改，避免完全重写；
3. 优先保留 SQL_1 和 SQL_2 中结构规范、语义清晰、逻辑准确的部分；
4. 如果其中一条 SQL 已满足用户意图，可直接输出；
5. 仅返回符合 JSON 格式的 SQL 查询语句，禁止添加任何说明性内容。
"""

# 修复路径3：CoT 融合修复的提示词
instruction_cot = """
你是一位精通 SQL 查询优化、语义理解与结构重构的专家，任务是基于用户查询意图，融合多轮 SQL 演进结果（SQL_1 ~ SQL_5），产出一个语义最准确、结构最合理、结果最可信的最终 SQL 查询语句。

### 核心任务：
1. 分析 SQL_1 ~ SQL_5 的演进过程与逻辑改动；
2. 找出每条 SQL 的正向贡献（准确表达的部分）和负面问题（错误或冗余）；
3. 基于 Chain-of-Thought（思维链）方法，融合它们的优点，规避错误；
4. 生成一条语义对齐用户意图、逻辑严谨、字段正确、结构清晰的 SQL 查询。

### 输入：
1. 用户问题：自然语言描述的查询意图；
2. SQL_1 ~ SQL_5：包括 SQL 查询语句、执行结果（如有）、执行失败时的错误信息、缺陷说明以及修复版本（SQL_3~SQL_5）的修复动机；
   - SQL_1：基于完整数据库结构信息生成，偏向结构完整性；
   - SQL_2：基于模式链接结果 + 列示例数据生成，偏向字段语义匹配；
   - SQL_3：融合 SQL_1 与 SQL_2 的初始合成版本；
   - SQL_4：对 SQL_3 的修复版本，可能引入新的逻辑偏差；
   - SQL_5：回退到 SQL_1 + SQL_2 再次融合得到；
3. 数据库结构信息：表名、字段名、字段类型、 表间关系等；
4. 字段解释与枚举值：帮助理解字段含义与可选值；
5. 相关列样本数据：每列的示例值、空值、格式异常等；
6. 与问题相关的领域知识（如有）：对业务背景的补充说明。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`）。
- 请仅以 JSON 格式返回最终融合优化后的 SQL 查询，格式如下：{"sql": "最终合成的 SQL 查询语句"}

### 操作步骤：
1. 分析 SQL_1：
    - 结构上是否存在字段冗余？
    - 是否字段意义不清？
2. 分析 SQL_2：
    - 是否字段选择精准？
    - 但结构连接、聚合是否不完整？
3. 分析 SQL_3（SQL_1 + SQL_2 的第一次融合）：
    - 是否正确融合了前两条 SQL？
    - 是否存在结构混乱、冗余、逻辑冲突或遗漏？
4.分析 SQL_4（对 SQL_3 的修复版本）：
    - 修复是否到位？是否脱离了用户原始意图？
    - 是否出现了新的字段、逻辑或表连接错误？
5. 分析 SQL_5（从 SQL_1 + SQL_2 回退重构）：
    - 是否更接近用户意图？
    - 是否规避了 SQL_3 与 SQL_4 的错误路径？
6. 多路径融合与结构重构：
    - 综合五条 SQL 的正向逻辑，复用有效结构；
    - 剔除冗余字段、不当逻辑与语法冲突；
    - 确保最终 SQL 的语义覆盖全面、结构合理、可正确执行。

### 注意事项
- 所有使用的字段名和表名必须严格来自于数据库结构信息，不允许使用未出现的字段或表。
- 禁止简单拼接五条 SQL，必须通过语义分析进行融合重构；
- 优先采纳 SQL_5 中的结构优化，结合 SQL_2 中的字段语义匹配；
- 若某条 SQL 已完全满足用户意图，可直接输出该 SQL；
- 输出格式必须为严格 JSON，不得附带任何注释、解释或过程说明。
"""

def extract_json(message):
    """
    尝试从 LLM 返回中提取 JSON 对象：
    1. 优先匹配 ```json { ... } ``` 代码块；
    2. 如果没有，再找第一个 '{' 和最后一个 '}' 之间的内容；
    3. 如果都失败，就原样返回。

    并且：
      - 如果提取到的字符串在闭合 '}' 前是 '";'，则把它替换为 '"}'
      - 否则，若在闭合 '}' 前缺少 '"'，再自动补上一个
    """
    def _cleanup(json_str: str) -> str:
        # —— 新增：如果闭合 } 前是 '";'，替换成 '"}'
        json_str = re.sub(r'";\s*}$', '"}', json_str)

        # —— 原有：补齐缺失的双引号
        if not json_str.endswith('}'):
            return json_str

        # 跳过 '}' 之前所有空白，找到最后一个非空白字符
        i = len(json_str) - 2
        while i >= 0 and json_str[i].isspace():
            i -= 1

        # 如果它不是 '"'，就在它后面插入一个
        if i >= 0 and json_str[i] != '"':
            insert_pos = i + 1
            json_str = json_str[:insert_pos] + '"' + json_str[insert_pos:]

        return json_str

    # 1. 尝试 ```json … ``` 代码块
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", message, re.DOTALL)
        if match:
            return _cleanup(match.group(1))
    except Exception as e:
        logging.error(f"Error extracting JSON with regex: {e}")

    # 2. 尝试最简单的 { … } 截取
    try:
        start = message.find("{")
        end   = message.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = message[start:end + 1]
            return _cleanup(candidate)
        else:
            logging.warning("未找到有效 JSON 区间，直接返回原始消息")
            return message
    except Exception as e:
        logging.error(f"Error extracting JSON using find: {e}")
        return message

def execute_single_sql(db_name, sql):
    try:
        row_count, column_count, result_preview, exec_time = execute_sql(sql, db_name)
    except Exception as e:
        logging.error(f"SQL 执行异常，数据库: {db_name}, SQL: {sql}. 错误: {e}")
        return {
            "sql": sql,
            "isvalid": False,
            "exec_time": 0,
            "error": f"exception: {e}"
        }
    if re.match(r"^(error:|timeouterror:)", result_preview, re.IGNORECASE):
        logging.info(f"SQL 执行失败: {sql}，错误信息: {result_preview}")
        return {
            "sql": sql,
            "isvalid": False,
            "exec_time": exec_time,
            "error": result_preview
        }
    else:
        logging.info(f"SQL 执行成功: {sql}")
        return {
            "sql": sql,
            "isvalid": True,
            "row_count": row_count,
            "column_count": column_count,
            "result_preview": result_preview,
            "exec_time": exec_time,
            "result": result_preview
        }

def build_context(question, schema, foreign_key, evidence, explanation, data, sql, result, reason_list):
    if len(reason_list) >= 2:
        sql_result = ""
        for i in range(len(sql)):
            sql_result += f'SQL: "{sql[i]}",'
            if result[i]["isvalid"]:
                sql_result += f'执行结果："{result[i]["result"]}",'
            else:
                sql_result += f'执行报错："{result[i]["error"]}",'
            if i > 2:
                sql_result += f'修复原因："{reason_list[i - 2]}"'
            sql_result += "\n"
    elif len(reason_list) == 1:
        sql_result = ""
        sql_result += f'SQL: "{sql[-1]}",'
        if result[-1]["isvalid"]:
            sql_result += f'执行结果："{result[-1]["result"]}",'
        else:
            sql_result += f'执行报错："{result[-1]["error"]}",'

        sql_result += f'修复原因："{reason_list[-1]}"\n'
    else:
        sql_result = ""
        sql_result += f'SQL: "{sql[0]}",'
        if result[0]["isvalid"]:
            sql_result += f'执行结果："{result[0]["result"]}",'
        else:
            sql_result += f'执行报错："{result[0]["error"]}",'
        sql_result += "\n"
        

    table_info = (
            f"Sqlite SQL 表及其属性：\n{schema}\n"
            f"Sqlite SQL 表的外键信息，用于表连接：\n{foreign_key}\n"
            f"每一列的含义：\n{explanation}\n"
        )
    
    context = (
            f'用户问题: \n"{question}"\n'
            f'当前 SQL 查询及其执行结果：\n{sql_result}\n'
            f'数据库结构信息：\n{table_info}\n'
            f'字段解释与枚举值：\n{explanation}\n'
            f'相关列的样本数据：\n{data}\n'
        )

    if evidence != "":
        context += f'与问题相关的领域知识：\n{evidence}\n'

    return context

# 判断是否需要修复
def needs_correction(question, schema, foreign_key, evidence, explanation, data, sql, result):
    if result['isvalid']:
        if len(result['result_preview']) == 0:
            return True, "Empty result"
        else:
            # 使用 LLM 判断语义偏离
            context = build_context(question, schema, foreign_key, evidence, explanation, data, [sql], [result], [])
            llm = QWEN_LLM_CODER()
            llm_judgment = llm(instruction_need, context)
            if "【是否修复】：需要修复" in llm_judgment:
                return True, llm_judgment
            return False, "SQL pass"
    else:
        return True, "SQL execution error"

# 修复路径1：直接修复 SQL_3
def fix_sql(question, schema, foreign_key, evidence, explanation, data, sql_3, result3, reason_list):
    try:
        # 使用 LLM 修复 SQL_3
        context = build_context(question, schema, foreign_key, evidence, explanation, data, [sql_3], [result3], reason_list)
        llm = QWEN_LLM_CODER()
        response = llm(instruction_fix, context)
        response = extract_json(response)
        try:
            response_json = json.loads(response)
            fixed_sql =  response_json.get("sql", "")
            return fixed_sql
        except json.JSONDecodeError as e:
            logging.warning(f"fix_sql 无法解析 LLM 返回的 JSON: {e}, 返回空字符串" + f"\n{response}")
            return ""
    except Exception as e:
        logging.error(f"直接修复 SQL_3 异常:{e}")

# 修复路径2：基于 SQL_1 / SQL_2 重构
def reconstruct_from_sql1_or_sql2(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list):
    try:
        context = build_context(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list)
        llm = QWEN_LLM_CODER()
        response = llm(instruction_reconstruct, context)
        response = extract_json(response)
        try:
            response_json = json.loads(response)
            reconstruct_sql =  response_json.get("sql", "")
            return reconstruct_sql
        except json.JSONDecodeError as e:
            logging.warning(f"reconstruct_from_sql1_or_sql2 无法解析 LLM 返回的 JSON: {e}, 返回空字符串" + f"\n{response}")
            return ""
    except Exception as e:
        logging.error(f"reconstruct_from_sql1_or_sql2 异常:{e}")

# 修复路径3：CoT 融合修复
def cot_fusion_fix(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list):
    try:
        context = build_context(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list)
        llm = QWEN_LLM_CODER()
        response = llm(instruction_cot, context)
        response = extract_json(response)
        try:
            response_json = json.loads(response)
            reconstruct_sql =  response_json.get("sql", "")
            return reconstruct_sql
        except json.JSONDecodeError as e:
            logging.warning(f"cot_fusion_fix 无法解析 LLM 返回的 JSON: {e}, 返回空字符串" + f"\n{response}")
            return ""
    except Exception as e:
        logging.error(f"cot_fusion_fix 异常:{e}")

def process_item(item):
    try:
        db = item.get("db")
        question = item.get("question")
        evidence = item.get("evidence")
        schema = item.get("schema")
        foreign_key = item.get("foreign_key")
        explanation = item.get("explanation")
        data = item.get("data")
        sql_1 = item.get("sql_1")
        sql_2 = item.get("sql_2")
        sql_3 = item.get("sql_3")

        entity = {
            "question_id": item.get("question_id"),
            "db": db,
            "question": question,
            "sql_1": sql_1,
            "sql_2": sql_2,
            "sql_3": sql_3,
            "sql_final": "",
            "evidence": evidence,
            "schema": schema,
            "foreign_key": foreign_key,
            "explanation": explanation,
            "data": data,
            "difficulty": item.get("difficulty")
        }

        sql_list = [sql_1, sql_2, sql_3]

        # 0. 执行 SQL_3
        result3 = execute_single_sql(db, sql_3)

        # 1. 判断是否需要修复
        needs_fix, reason = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_3, result3)
        if not needs_fix:
            entity['sql_final'] = sql_3
            entity['count'] = 3
            return entity

        ## 需要修复
        print(f"sql_3 修复触发：{reason}")

        reason_list = [reason]
        
        # 2. 修复调度器
        # 路径 1：直接修 SQL_3
        sql_4 = fix_sql(question, schema, foreign_key, evidence, explanation, data, sql_3, result3, reason_list)
        # 验证这条新生成的sql语句是否有效
        result4 = execute_single_sql(db, sql_4)
        need_fix_1, reason_1 = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_4, result4)
        entity['sql_4'] = sql_4
        if not need_fix_1:
            entity['sql_final'] = sql_4
            entity['count'] = 4
            return entity
        
        print(f"sql_4 修复触发：{reason_1}")

        sql_list.append(sql_4)
        result1 = execute_single_sql(db, sql_1)
        result2 = execute_single_sql(db, sql_2)
        result_list = [result1, result2, result3, result4]
        reason_list.append(reason_1)
        # 路径 2：基于 SQL_1 / SQL_2 重构
        sql_5 = reconstruct_from_sql1_or_sql2(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list)
        # 验证这条新生成的sql语句是否有效
        result5 = execute_single_sql(db, sql_5)
        need_fix_2, reason_2 = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_5, result5)
        entity['sql_5'] = sql_5
        if not need_fix_2:
            entity['sql_final'] = sql_5
            entity['count'] = 5
            return entity
        
        print(f"sql_5 修复触发：{reason_2}")

        sql_list.append(sql_5)
        result_list.append(result5)
        reason_list.append(reason_2)
        # 路径 3：CoT 融合修复
        sql_6 = cot_fusion_fix(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list)
        # 验证这条新生成的sql语句是否有效
        result6 = execute_single_sql(db, sql_6)
        need_fix_3, reason_3 = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_6, result6)
        entity['sql_6'] = sql_6
        if not need_fix_3:
            entity['sql_final'] = sql_6
            entity['count'] = 6
            return entity
        
        print(f"sql_6 修复触发：{reason_3}")

        # 按从后往前检查：sql_6 > sql_5 > sql_4
        if result6.get("isvalid", False):
            entity["sql_final"] = sql_6
        elif result5.get("isvalid", False):
            entity["sql_final"] = sql_5
        elif result4.get("isvalid", False):
            # 如果 sql_4 的执行结果有效则用 sql_4，否则仍然保持 sql_3（因为最开始就预设了 sql_4=sql_3）
            entity["sql_final"] = sql_4 if result4.get("isvalid", False) else sql_3
        else:
            entity["sql_final"] = sql_6  # 若其他均失败，则返回最新结果
        entity['count'] = 7
        return entity
    except KeyError as e:
        logging.warning(f"缺少键 {e}，item={item}")
    except Exception as e:
        logging.error(f"处理 item 时异常: {e}")
    return None

def main(input_file, output_file, start_index, max_workers=8):
    items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"解码 JSON 行时错误: {e}")
    except Exception as e:
        logging.error(f"读取文件 {input_file} 异常: {e}")
        return

    items_to_process = items[start_index:]
    logging.info(f"待处理条目数: {len(items_to_process)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        futures = {executor.submit(process_item, it): it for it in items_to_process}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing items"):
            try:
                result = future.result()
                if result:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
            except Exception as e:
                logging.error(f"处理并发任务时异常: {e}")

    logging.info(f"已完成，结果保存在 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type = int, default = 0)
    parser.add_argument("--input_file", type = str, default = "src/dataset/qwen/coder-7b/3_cot_synthesize_sql.jsonl")
    parser.add_argument("--output_file", type = str, default = "src/dataset/qwen/coder-7b/4_final_sql.jsonl")
    parser.add_argument("--max_workers", type = int, default = 8, help = "线程数")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.start_index, args.max_workers)