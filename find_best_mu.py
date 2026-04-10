import pandas as pd
import numpy as np

# Load data
df1 = pd.read_csv('parameter/mu_search_all_ETTh1.csv')
df2 = pd.read_csv('parameter/mu_search_all_IDF_OilTemp.csv')

# Drop NA / failed runs
df1 = df1.dropna(subset=['perf_improvement', 'mse'])
df2 = df2.dropna(subset=['perf_improvement', 'mse'])

# Round to merge properly
df1['mu_1'] = df1['mu_1'].round(2)
df1['mu_2'] = df1['mu_2'].round(2)
df1['mu_3'] = df1['mu_3'].round(2)
df1['mu_4'] = df1['mu_4'].round(2)

df2['mu_1'] = df2['mu_1'].round(2)
df2['mu_2'] = df2['mu_2'].round(2)
df2['mu_3'] = df2['mu_3'].round(2)
df2['mu_4'] = df2['mu_4'].round(2)

# Merge
merged = pd.merge(df1, df2, on=['mu_1', 'mu_2', 'mu_3', 'mu_4'], suffixes=('_etth1', '_idf'))

# Sort by some joint metric
# We want high perf, low mse.
# Let's normalize them to 0-1 range to create a joint score, or just print the top Pareto points.
merged['perf_sum'] = merged['perf_improvement_etth1'] + merged['perf_improvement_idf']
merged['mse_sum'] = merged['mse_etth1'] + merged['mse_idf']

# Print top by perf_sum
print("Top 5 by Combined Performance Improvement:")
print(merged.sort_values('perf_sum', ascending=False)[['mu_1', 'mu_2', 'mu_3', 'mu_4', 'perf_improvement_etth1', 'perf_improvement_idf', 'mse_etth1', 'mse_idf']].head(5).to_string(index=False))

print("\nTop 5 by Lowest Combined MSE (while keeping positive perf):")
positive_perf = merged[(merged['perf_improvement_etth1'] > 0) & (merged['perf_improvement_idf'] > 0)]
print(positive_perf.sort_values('mse_sum', ascending=True)[['mu_1', 'mu_2', 'mu_3', 'mu_4', 'perf_improvement_etth1', 'perf_improvement_idf', 'mse_etth1', 'mse_idf']].head(10).to_string(index=False))
