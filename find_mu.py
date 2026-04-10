import pandas as pd
df = pd.read_csv('/home/yyy/TSC/TSClean/AutoClean/parameter/lambda_search_all_ETTh1.csv')
for i in range(1, 4):
    for base in [0.2, 0.1]:
        print(f"Col lambda_{i}, base={base}")
        sub = df[df[f'lambda_{i}'] == base]
        if not sub.empty:
            best = sub.sort_values(by='perf_improvement', ascending=False).iloc[0]
            p0 = best['perf_improvement']
            m0 = best['mse']
            print(f"  Best at {base}: {best.to_dict()}")
