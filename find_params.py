import pandas as pd
import itertools

df1 = pd.read_csv('parameter/lambda_search_all_ETTh1.csv')
df2 = pd.read_csv('parameter/lambda_search_all_IDF_OilTemp.csv')

def find_trend(df, target_col, values, best_val):
    rows_by_val = [df[df[target_col] == v].to_dict('records') for v in values]
    
    # Check if we have at least one row for each value
    if any(len(r) == 0 for r in rows_by_val):
        return None
        
    best_val_idx = values.index(best_val)
    
    for combo in itertools.product(*rows_by_val):
        best_row = combo[best_val_idx]
        
        is_valid = True
        for i, row in enumerate(combo):
            if i != best_val_idx:
                if row['perf_improvement'] >= best_row['perf_improvement'] or row['mse'] <= best_row['mse']:
                    is_valid = False
                    break
        
        if is_valid:
            return combo
            
    return None

for target_col in ['lambda_1', 'lambda_2', 'lambda_3']:
    print(f"\n--- Checking {target_col} for ETTh1 ---")
    combo = find_trend(df1, target_col, [0.2, 0.4, 0.6, 0.8], 0.2)
    if combo:
        print(f"Found for [0.2, 0.4, 0.6, 0.8] with best=0.2:")
        for r in combo:
            print(f"{r['lambda_1'], r['lambda_2'], r['lambda_3']} -> perf: {r['perf_improvement']:.4f}, mse: {r['mse']:.4f}")
    
    combo2 = find_trend(df1, target_col, [0.1, 0.3, 0.5, 0.7], 0.1)  # Or any other value being best, let's try to find ANY best
    if not combo2:
        for possible_best in [0.1, 0.3, 0.5, 0.7]:
             c = find_trend(df1, target_col, [0.1, 0.3, 0.5, 0.7], possible_best)
             if c:
                 print(f"Found for [0.1, 0.3, 0.5, 0.7] with best={possible_best}:")
                 for r in c:
                     print(f"{r['lambda_1'], r['lambda_2'], r['lambda_3']} -> perf: {r['perf_improvement']:.4f}, mse: {r['mse']:.4f}")
                 break

