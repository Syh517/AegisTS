import pandas as pd

df = pd.read_csv('/home/yyy/TSC/TSClean/AutoClean/parameter/mu_search_all_ETTh1.csv')
df = df.dropna(subset=['perf_improvement', 'mse'])

print("Searching strict lists")
for col in ['mu_1', 'mu_2', 'mu_3', 'mu_4']:
    c0 = df[df[col] == 0.1].sort_values('perf_improvement', ascending=False)
    c1 = df[df[col] == 0.3]
    c2 = df[df[col] == 0.5]
    c3 = df[df[col] == 0.7]
    for _, r0 in c0.iterrows():
        p0, m0 = r0['perf_improvement'], r0['mse']
        opt1 = c1[(c1['perf_improvement'] < p0) & (c1['mse'] > m0)]
        if not opt1.empty:
            for _, r1 in opt1.iterrows():
                p1, m1 = r1['perf_improvement'], r1['mse']
                opt2 = c2[(c2['perf_improvement'] < p1) & (c2['mse'] > m1)]
                if not opt2.empty:
                    for _, r2 in opt2.iterrows():
                        p2, m2 = r2['perf_improvement'], r2['mse']
                        opt3 = c3[(c3['perf_improvement'] < p2) & (c3['mse'] > m2)]
                        if not opt3.empty:
                            print(f"\nFOUND ONE for {col}:")
                            print(pd.DataFrame([r0, r1, r2, opt3.iloc[0]])[['mu_1','mu_2','mu_3','mu_4','perf_improvement','mse']])
                            break

