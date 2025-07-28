import os
import json
import pandas as pd
print("当前工作目录:", os.getcwd())

# 构造基础目录和输出目录（跨平台拼接路径）
base_dir = os.path.join(os.getcwd(), 'database', 'dev_databases')
output_base = os.path.join(os.getcwd(), 'milvus', 'schema1')

# 如果输出目录不存在则创建
if not os.path.exists(output_base):
    os.makedirs(output_base)

if os.path.exists(base_dir):
    # 遍历 base_dir 下每个条目（目录）
    entries = os.listdir(base_dir)
    for entry in entries:
        # 忽略系统文件（例如 ".DS_Store"）
        if entry == "student_club":
            print("处理目录:", entry)
            # 构造子目录路径，例如：base_dir/entry/database_description
            dir1 = os.path.join(base_dir, entry, 'database_description')
            if os.path.exists(dir1):
                files = os.listdir(dir1)
                results = {}  # 用字典存储每个 CSV 文件转换后的 JSON 数据
                for file in files:
                    # 过滤系统文件以及非 CSV 文件
                    if file != ".DS_Store" and file.lower().endswith('.csv'):
                        file_path = os.path.join(dir1, file)
                        try:
                            # 读取 CSV 文件
                            df = pd.read_csv(file_path, encoding='utf-8')
                        except Exception as e:
                            print(f"读取 {file_path} 时出错: {e}")
                            continue
                        
                        # 转换为 JSON 格式
                        # 将 DataFrame 转成 list of dicts
                        records = df.to_dict(orient='records')

                        # 遍历每条记录，填充空值
                        for rec in records:
                            orig = rec.get("original_column_name", "")
                            for field in ("column_name", "column_description", "value_description"):
                                val = rec.get(field)
                                # 如果是 NaN 或者 空字符串，就用 orig 覆盖
                                if pd.isna(val) or val == "":
                                    rec[field] = orig
                                else:
                                    orig = val
                        
                        # 从文件名中去除扩展名作为名称
                        name = os.path.splitext(file)[0]
                        results[name] = records
                        
                # 构造输出文件路径，文件名用 entry 命名，如 <entry>.json
                output_file = os.path.join(output_base, f"{entry}.json")
                # 将 results 字典写入 JSON 文件，可以使用 json.dump 将其转换为 JSON 格式
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    # 如果希望保存为 JSON 格式的文件，建议将 results 转成 JSON 字符串
                    json.dump(results, json_file, ensure_ascii=False, indent=4)
                print(f"已生成文件: {output_file}")
            else:
                print(f"路径 {dir1} 不存在。")
else:
    print(f"目录 {base_dir} 不存在，请确认路径是否正确。")