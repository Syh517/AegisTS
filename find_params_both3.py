import pandas as pd
import itertools

df1 = pd.read_csv('parameter/lambda_search_all_ETTh1.csv')
df2 = pd.read_csv('parameter/lambda_search_all_IDF_OilTemp.csv')

def score_pair(rowA, rowB, bestA, bestB):
    # Returns 1 if valid, 0 if not
    if rowA['perf_improvement'] >= bestA['perf_improvement'] or rowA['mse'] <= bestA['mse']:
        return 0
    if rowB.iloc[0]['perf_improvement'] >= bestB['perf_improvement'] or rowB.iloc[0]['mse'] <= bestB['mse']:
        return 0
    return 1

# Try to find the closest...
