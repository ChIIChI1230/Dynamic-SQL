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

# CoT 融合修复的提示词
instruction_cot = """
你是一位精通 SQL 查询优化、语义理解与结构重构 的专家。你的任务是基于用户提出的自然语言查询意图，采用 Chain-of-Thought（思维链）多路径融合方法，对用户提供的多个 SQL 版本进行系统分析与优化，逐步改进并最终生成一条语义最精准、逻辑最严谨、结构最清晰、执行结果最可信的 SQL 查询语句。

### 核心任务：
1. 全面分析 SQL 语句的演化过程与结构逻辑变迁；
2. 精确识别每个 SQL 的正向贡献（合理表达）与负向问题（错误、冗余、语义偏差等）；
3. 基于 CoT 思维链方式，融合优点、规避缺陷、迭代演进；
4. 生成一条语义对齐原始意图、结构最优、可正确执行的最终 SQL 查询。

### 输入：
1. 用户问题：自然语言描述的查询意图；
2. SQL 版本信息：  
   - 初始版本：SQL_1 与 SQL_2 （参考不同方案），  
   - 第一版融合生成 SQL_3（基于 SQL_1 与 SQL_2 的初步整合结果）；  
   - 后续融合版本（例如 SQL_4、SQL_5...）及每版执行结果、错误信息、修复原因说明；
3. 数据库结构信息：表名、字段名、字段类型、 表间关系等；
4. 字段解释与枚举值：帮助理解字段含义与可选值；
5. 相关列样本数据：每列的示例值、空值、格式异常等；
6. 与问题相关的领域知识（如有）：对业务背景的补充说明。

### 输出：
- 在 SQL 语句中所有表名和字段名均使用反引号括起来（如 `table_name`、`column_name`);
- SQL 语句中字段名和 SQL 表达式直接必须有空格，避免 SQL 无法运行;
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
4.分析 SQL_4、SQL_5 等后续版本：
    - 修复是否有效？是否引入新问题或偏离原意？
    - 是否更接近用户预期目标？
    - 是否彻底规避了早期版本的所有路径性错误？
5. 多路径融合与结构重构：
    - 重建结构框架，提取各版本有效组件；
    - 严格剔除语义模糊、结构冗余或语法冲突部分；
    - 生成一条符合原始意图、字段正确、逻辑自洽的最终 SQL 查询。

### 错误字段及执行异常处理规则
1. 当某个 SQL 版本执行时报错提示 “列不存在”（如 unknown column、invalid column name、no such column等），必须：
    - 回查提供的字段名列表（数据库结构信息 + 字段解释）；
    - 确认正确字段名并进行替换；
    - 不得凭空臆造字段名称；
    - 修复后的 SQL 必须重新验证结构和语义正确性后才能参与最终融合。
2. 当 SQL 报错为 执行超时（TimeoutError），必须：
    - 应检查是否存在复杂嵌套、多重 JOIN、无过滤条件或未索引列排序等；
    - 必须在保持语义正确的前提下进行结构重构或加条件剪枝，优化执行效率。

### 注意事项
- 所有使用的字段名和表名必须严格来自于数据库结构信息，不允许使用未出现的字段或表。
- 不允许仅简单拼接多个 SQL 版本，而必须通过深入的语义和逻辑分析进行融合重构；
- 输出格式必须严格为 JSON，不得包含任何注释、解释或额外说明。
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
        logging.info(f"SQL 执行失败: {sql}, 错误信息: {result_preview}")
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
    if len(reason_list) > 0:
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
    else:
        sql_result = ""
        for i in range(len(sql)):
            sql_result += f'SQL: "{sql[i]}",'
            if result[i]["isvalid"]:
                sql_result += f'执行结果："{result[i]["result"]}",'
            else:
                sql_result += f'执行报错："{result[i]["error"]}",'
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
        print("result:" +  result['result_preview'])
        if len(result['result_preview']) == 0:
            return True, "Empty result"
        else:
            # # 使用 LLM 判断语义偏离
            # context = build_context(question, schema, foreign_key, evidence, explanation, data, [sql], [result], [])
            # llm = QWEN_LLM_CODER()
            # llm_judgment = llm(instruction_need, context)
            # if "【是否修复】：需要修复" in llm_judgment:
            #     return True, llm_judgment
            
            return False, "SQL pass"
    else:
        return True, "SQL execution error"

# CoT 融合修复
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
            "sql_final": "",
            "sql_1": sql_1,
            "sql_2": sql_2,
            "sql_3": sql_3,
            "sql_4": "",
            "sql_5": "",
            "sql_6": "",
            "evidence": evidence,
            "difficulty": item.get("difficulty")
        }
        
        result3 = execute_single_sql(db, sql_3)
        needs_fix, reason = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_3, result3)
        
        # if needs_fix:
        #     entity['sql_final'] = sql_3
        #     entity['reason'] = reason
        #     entity['error'] = result3.get("error", "") 
        #     return entity
        
        if not needs_fix:
            entity['sql_final'] = sql_3
            entity['count'] = 3
            return entity
        
        sql_list = [sql_1, sql_2, sql_3]
        result1 = execute_single_sql(db, sql_1)
        result2 = execute_single_sql(db, sql_2)
        result_list = [result1, result2, result3]
        reason_list = [reason]
        count = 0
        sql_final = ""
        while needs_fix and count < 3:
            sql_final = cot_fusion_fix(question, schema, foreign_key, evidence, explanation, data, sql_list, result_list, reason_list)
            # 验证这条新生成的sql语句是否有效
            result = execute_single_sql(db, sql_final)
            need_fix, reason = needs_correction(question, schema, foreign_key, evidence, explanation, data, sql_final, result)
            entity[f'sql_{count + 4}'] = sql_final
            if not need_fix:
                entity['sql_final'] = sql_final
                entity['count'] = count + 4
                return entity
            sql_list.append(sql_final)
            reason_list.append(reason)
            result_list.append(result)
            count += 1

        for i in range(len(result_list) - 1, -1, -1):
            if result_list[i]['isvalid']:
                sql_final = sql_list[i]
                break
        entity['sql_final'] = sql_final
        entity['count'] = 7
        entity['error'] = reason + "=========" + result['error'] if result['isvalid'] == False else ""
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
    parser.add_argument("--input_file", type = str, default = "src/dataset/qwen/coder-32b/en/3_cot_synthesize_sql1.jsonl")
    parser.add_argument("--output_file", type = str, default = "src/dataset/qwen/coder-32b/en/4_final_sql1.jsonl")
    parser.add_argument("--max_workers", type = int, default = 8, help = "线程数")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.start_index, args.max_workers)