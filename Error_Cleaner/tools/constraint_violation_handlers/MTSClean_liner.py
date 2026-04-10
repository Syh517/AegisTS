import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus
from typing import Dict, Any, List

class MTSClean:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def clean(self, 
              data: pd.DataFrame, 
              speed_constraints: Dict[str, Any] = None, 
              acceleration_constraints: Dict[str, Any] = None, 
              row_constraints: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        核心修复逻辑：
        1. 目标函数：minimize sum |x_new - x_old|
        2. 约束条件：
           - 满足速度、加速度范围（Column Constraints）
           - 满足变量间回归方程残差范围（Row Constraints）
        """
        df = data.copy()
        T, N = df.shape
        cols = df.columns.tolist()
        
        # 定义线性规划问题
        prob = LpProblem("MTSClean_Repair", LpMinimize)
        
        # 1. 创建变量：每个数据点对应的修复值 x_new
        # x_vars[col][t]
        x_vars = {col: [LpVariable(f"x_{col}_{t}") for t in range(T)] for col in cols}
        
        # 2. 创建辅助变量用于处理目标函数中的绝对值: |x_new - x_old| <= d
        # minimize sum(d)
        diff_vars = {col: [LpVariable(f"d_{col}_{t}", lowBound=0) for t in range(T)] for col in cols}
        
        # 目标函数：最小化总修改量
        prob += lpSum(diff_vars[c][t] for c in cols for t in range(T))

        # 3. 添加绝对值约束： x_new - x_old <= d 且 x_old - x_new <= d
        for c in cols:
            for t in range(T):
                prob += x_vars[c][t] - df.iloc[t][c] <= diff_vars[c][t]
                prob += df.iloc[t][c] - x_vars[c][t] <= diff_vars[c][t]

        # 4. 添加列约束 (Column Constraints: 速度与加速度)
        # 根据你提供的格式：{'col_1': (min_val, max_val)}
        if speed_constraints:
            for col, bounds in speed_constraints.items():
                if col in x_vars:
                    v_min, v_max = bounds[0], bounds[1] # 通常输入是对称阈值
                    for t in range(1, T):
                        prob += x_vars[col][t] - x_vars[col][t-1] <= v_max
                        prob += x_vars[col][t] - x_vars[col][t-1] >= v_min

        if acceleration_constraints:
            for col, bounds in acceleration_constraints.items():
                if col in x_vars:
                    a_min, a_max = bounds[0], bounds[1]
                    for t in range(2, T):
                        # 二阶差分: (x_t - x_t-1) - (x_t-1 - x_t-2)
                        prob += x_vars[col][t] - 2*x_vars[col][t-1] + x_vars[col][t-2] <= a_max
                        prob += x_vars[col][t] - 2*x_vars[col][t-1] + x_vars[col][t-2] >= a_min

        # 5. 添加行约束 (Row Constraints: 变量间线性关系)
        # 格式：[{'target': 'col_1', 'model_terms': {'col_3': 1.0...}, 'intercept': 0.12, 'lower': -0.6, 'upper': 0.6}]
        if row_constraints:
            for rule in row_constraints:
                target = rule['target']
                features = rule['model_terms']
                intercept = rule['intercept']
                l_bound = rule['lower']
                u_bound = rule['upper']
                
                if target in x_vars:
                    for t in range(T):
                        # 构建线性组合：target_val - sum(w_i * feature_i)
                        expr = x_vars[target][t] - lpSum(coeff * x_vars[feat][t] for feat, coeff in features.items() if feat in x_vars)
                        # 约束：residual + intercept 落在 [lower, upper]
                        # 即：target - sum(w*feat) 落在 [intercept+lower, intercept+upper]
                        prob += expr <= (intercept + u_bound)
                        prob += expr >= (intercept + l_bound)

        # 6. 求解
        solver = PULP_CBC_CMD(msg=self.verbose)
        status = prob.solve(solver)

        if LpStatus[status] == 'Optimal':
            # 将结果写回 DataFrame
            for c in cols:
                for t in range(T):
                    df.iloc[t, cols.index(c)] = x_vars[c][t].varValue
            return df
        else:
            if self.verbose: print(f"Warning: Solver status is {LpStatus[status]}")
            return data

# 对接函数
def _clean_with_mtclean(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用MTSClean清洗数据"""
    try:
        cleaner = MTSClean()
        cleaned_data = cleaner.clean(
            data,
            speed_constraints=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            row_constraints=constraints.get('row_constraints', [])
        )
        return cleaned_data
    except Exception as e:
        print(f"使用MTSClean清洗数据时出错: {e}")
        return data
    

if __name__ == "__main__":

    # --- 1. 准备模拟数据 (包含明显的脏数据) ---
    data = {
        'col_1': [10.1, 10.2, 15.0, 10.4, 10.5], # 15.0 是一个脏跳变
        'col_3': [5.0, 5.1, 5.2, 5.3, 5.4],
        'col_5': [5.0, 5.0, 5.1, 5.1, 5.2]
    }
    raw_df = pd.DataFrame(data)

    # --- 2. 准备你的约束格式 ---
    my_constraints = {
        'row_constraints': [
            {
                'target': 'col_1',
                'model_terms': {'col_3': 1.0, 'col_5': 1.0},
                'intercept': 0.127,
                'lower': -0.63,
                'upper': 0.66
            }
        ],
        'speed_constraints': {
            'col_1': (0, 2.0), # col_1 的速度限制为 2.0
        },
        'accel_constraints': {
            'col_1': (0, 2.0), # col_1 的加速度限制为 2.0
        }
    }

    # --- 3. 调用刚才封装的方法 ---
    print("--- 原始数据 (存在脏数据 col_1=15.0) ---")
    print(raw_df)

    # 这里调用我们之前定义的封装函数
    cleaned_df = _clean_with_mtclean(raw_df, my_constraints)

    print("\n--- 清洗后的数据 ---")
    print(cleaned_df.round(2))

    # --- 4. 验证效果 ---
    # 检查 col_1 在索引 2 的位置是否被拉回
    # 按照约束：col_1 应该接近 col_3 + col_5 + 0.127 = 5.2 + 5.1 + 0.127 = 10.427
    # 且满足速度限制 (与前一个点 10.2 的差值不超过 2.0)