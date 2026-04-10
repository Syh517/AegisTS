import pandas as pd

df1 = pd.read_csv('parameter/lambda_search_all_ETTh1.csv')
df2 = pd.read_csv('parameter/lambda_search_all_IDF_OilTemp.csv')

combos = [
    # lambda_2 varied [0.2, 0.4, 0.6, 0.8]
    (0.3, 0.2, 0.5),
    (0.1, 0.4, 0.5),
    (0.3, 0.6, 0.1),
    (0.1, 0.8, 0.1)
]

for l1, l2, l3 in combos:
    r1 = df1[(df1.lambda_1.between(l1-0.01, l1+0.01)) & (df1.lambda_2.between(l2-0.01, l2+0.01))].iloc[0]
    r2 = df2[(df2.lambda_1.between(l1-0.01, l1+0.01)) & (df2.lambda_2.between(l2-0.01, l2+0.01))].iloc[0]
    print(f"L2={l2}: ETTh1(perf={r1.perf_improvement:.4f}, mse={r1.mse:.4f})  IDF(perf={r2.perf_improvement:.4f}, mse={r2.mse:.4f})")

print("---")
combos2 = [
    (0.2, 0.1, 0.7),
    (0.1, 0.3, 0.6),
    (0.3, 0.5, 0.2),
    (0.1, 0.7, 0.2)
]
for l1, l2, l3 in combos2:
    r1 = df1[(df1.lambda_1.between(l1-0.01, l1+0.01)) & (df1.lambda_2.between(l2-0.01, l2+0.01))].iloc[0]
    r2 = df2[(df2.lambda_1.between(l1-0.01, l1+0.01)) & (df2.lambda_2.between(l2-0.01, l2+0.01))].iloc[0]
    print(f"L2={l2}: ETTh1(perf={r1.perf_improvement:.4f}, mse={r1.mse:.4f})  IDF(perf={r2.perf_improvement:.4f}, mse={r2.mse:.4f})")

