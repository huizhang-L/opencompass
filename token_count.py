import os
import json
import pandas as pd
from transformers import AutoTokenizer

# 配置分词器（替换成你需要的分词器）
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/models/pretrain_models/QwQ-32B-Preview")

def process_json_file(filepath):
    """
    处理单个 JSON 文件，统计每个字段的 `prediction` 长度
    """
    with open(filepath, "r") as file:
        data = json.load(file)
    
    # 遍历 JSON 数据，计算每个字段的长度
    lengths = []
    for key, value in data.items():
        prediction = value.get("prediction", "")
        token_count = len(tokenizer.encode(prediction, add_special_tokens=False))
        lengths.append(token_count)
    
    return lengths

def process_all_json_files(directory):
    """
    批量处理路径下所有 JSON 文件，统计长度和平均值
    """
    all_lengths = []  # 保存所有文件中 `prediction` 的长度

    # 遍历目录下的所有 JSON 文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            lengths = process_json_file(filepath)
            all_lengths.extend(lengths)  # 收集所有文件的长度

    # 转换为 Pandas DataFrame
    df = pd.DataFrame({
        "lengths": all_lengths
    })

    # 计算平均长度
    avg_length = df["lengths"].mean()

    return df, avg_length

# 指定 JSON 文件所在的目录和保存路径
directory_path = "/mnt/data/user/lv_huijie/o1_overthink/opencompass/outputs/eval_math_test/20250118_051240/predictions/vllm-serve-qwen2.5-32b-instruct"  # 替换为你的目录路径
output_csv_path = "/mnt/data/user/lv_huijie/o1_overthink/opencompass/outputs/eval_math_test/20250118_051240/summary/token_num.csv"  # 替换为保存 CSV 的路径

# 处理所有 JSON 文件
df_lengths, avg_length = process_all_json_files(directory_path)

# 保存结果为 CSV 文件
df_lengths.to_csv(output_csv_path, index=False)

# 打印结果
print("每个字段的长度已保存到 CSV 文件：", output_csv_path)
print("\n所有字段的平均长度：")
print(f"平均长度: {avg_length:.2f}")