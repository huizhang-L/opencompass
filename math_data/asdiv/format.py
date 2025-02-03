import pandas as pd

# 1. 读取 JSONL 文件到 DataFrame
df = pd.read_json('/mnt/data/user/lv_huijie/o1_overthink/opencompass/math_data/asdiv/asdiv.jsonl', lines=True)

# 2. 筛选 'grade' 字段为 6 的行
filtered_df = df[df['Grade'] == 6]

# 3. 将筛选后的 DataFrame 保存为新的 JSONL 文件
filtered_df.to_json('/mnt/data/user/lv_huijie/o1_overthink/opencompass/math_data/asdiv/asdiv_6.jsonl', orient='records', lines=True)