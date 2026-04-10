import re
import pandas as pd

# 示例文本
text = """
mse: 0.0016
mnad: 0.0150
rra: 0.2716
precision: 0.2915
recall: 0.6547
mae_error: 0.0059
time_cost: 4751.7391
f1_score: 0.4033
"""

# 使用正则表达式提取数据
pattern = r"(\w+): ([\d\.]+)"
matches = re.findall(pattern, text)

# 将提取的数据转换为字典
data = {key: float(value) for key, value in matches}

# 特别处理 Evaluation Result 部分
eval_pattern = r"Evaluation Result:  {([^}]*)}"
eval_matches = re.search(eval_pattern, text)
if eval_matches:
    eval_text = eval_matches.group(1)
    eval_pairs = eval_text.split(", ")
    for pair in eval_pairs:
        key, value = pair.split(": ")
        # 去掉键名中的引号
        key = key.strip().strip("'")
        data[key] = float(value.strip())


# 打印提取的数据
print(data)


# 将数据转换为 DataFrame
df = pd.DataFrame([data])  # 使用列表包裹字典，使其成为一行

# 保存到 CSV 文件
df.to_csv("./results.csv", index=False)
