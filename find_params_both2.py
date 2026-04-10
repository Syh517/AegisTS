import pandas as pd
import itertools

df1 = pd.read_csv('parameter/lambda_search_all_ETTh1.csv')
df2 = pd.read_csv('parameter/lambda_search_all_IDF_OilTemp.csv')

def find_trend_both_any_best(dfA, dfB, target_col, values):
    rows_by_val = []
    
    for v in values:
        subset = dfA[dfA[target_col] == v]
        records = []
        for _, rowA in subset.iterrows():
            l1, l2 = round(rowA['lambda_1'], 2), round(rowA['lambda_2'], 2)
            rowB = dfB[(dfB.lambda_1.round(2) == l1) & (dfB.lambda_2.round(2) == l2)]
            if len(rowB) > 0:
                records.append({
                    'lambda_1': l1, 'lambda_2': l2, 'lambda_3': round(rowA['lambda_3'], 2),
                    'perfA': rowA['perf_improvement'], 'mseA': rowA['mse'],
                    'perfB': rowB.iloc[0]['perf_improvement'], 'mseB': rowB.iloc[0]['mse']
                })
        rows_by_val.append(records)
        
    if any(len(r) == 0 for r in rows_by_val):
        return None
        
    for best_val_idx, best_val in enumerate(values):
        for combo in itertools.product(*rows_by_val):
            best_row = combo[best_val_idx]
            
            is_valid = True
            for i, row in enumerate(combo):
                if i != best_val_idx:
                    if row['perfA'] >= best_row['perfA'] or row['mseA'] <= best_row['mseA']:
                        is_valid = False; break
                    if row['perfB'] >= best_row['perfB'] or row['mseB'] <= best_row['mseB']:
                        is_valid = False; break
            
            if is_valid:
                return combo, best_val
    return None, None

for target_col in ['lambda_1', 'lambda_2', 'lambda_3']:
    combo, best = find_trend_both_any_best(df1, df2, target_col, [0.2, 0.4, 0.6, 0.8])
    if combo:
        print(f"--- {target_col} [0.2, 0.4, 0.6, 0.8] best={best} ---")
        for r in combo:
            print(f"{r['lambda_1'], r['lambda_2'], r['lambda_3']} ETTh1(p:{r['perfA']:.4f}, m:{r['mseA']:.4f})  IDF(p:{r['perfB']:.4f}, m:{r['mseB']:.4f})")

