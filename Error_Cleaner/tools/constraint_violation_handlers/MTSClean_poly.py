import pandas as pd
import numpy as np
from pyomo.environ import *
from typing import List, Dict, Any
import re

class MTSClean:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _parse_term(self, key: str, model_vars_at_t: dict):
        """解析多项式项，例如 'col_1^2' 或 'col_3 col_1'"""
        if ' ' in key:
            parts = key.split(' ')
            expr = 1
            for p in parts:
                expr *= self._parse_term(p.strip(), model_vars_at_t)
            return expr
        
        if '^' in key:
            base, expo = key.split('^')
            return model_vars_at_t[base.strip()] ** float(expo)
        
        return model_vars_at_t[key.strip()]

    def clean(self, 
        data: pd.DataFrame, 
        speed_constraints: Dict[str, Any] = None, 
        acceleration_constraints: Dict[str, Any] = None,
        row_constraints: List[Dict[str, Any]] = None) -> pd.DataFrame:
        
        T, N = data.shape
        cols = data.columns.tolist()
        
        model = ConcreteModel()
        model.T = RangeSet(0, T - 1)
        model.N = Set(initialize=cols)
        
        # 决策变量初始化
        model.x = Var(model.N, model.T, initialize=lambda m, n, t: data.iloc[t, cols.index(n)])

        # 目标函数：最小化修复值与原始值的平方偏差
        def obj_rule(m):
            return sum((m.x[n, t] - data.iloc[t, cols.index(n)])**2 for n in m.N for t in m.T)
        model.obj = Objective(rule=obj_rule, sense=minimize)

        # 1. 速度约束 (完全对齐你的逻辑)
        if speed_constraints:
            model.speed_cons = ConstraintList()
            for col, bounds in speed_constraints.items():
                if col in cols:
                    v_min, v_max = bounds[0], bounds[1]
                    for t in range(1, T):
                        # x_t - x_t-1 落在 [v_min, v_max]
                        model.speed_cons.add(expr=inequality(v_min, model.x[col, t] - model.x[col, t-1], v_max))

        # 2. 加速度约束 (完全对齐你的逻辑)
        if acceleration_constraints:
            model.accel_cons = ConstraintList()
            for col, bounds in acceleration_constraints.items():
                if col in cols:
                    a_min, a_max = bounds[0], bounds[1]
                    for t in range(2, T):
                        # 二阶差分: x_t - 2*x_t-1 + x_t-2 落在 [a_min, a_max]
                        accel_expr = model.x[col, t] - 2*model.x[col, t-1] + model.x[col, t-2]
                        model.accel_cons.add(expr=inequality(a_min, accel_expr, a_max))

        # 3. 多项式行约束 (处理 target, model_terms, intercept 等)
        if row_constraints:
            model.row_cons = ConstraintList()
            for rule in row_constraints:
                target = rule['target']
                intercept = rule['intercept']
                l_bound, u_bound = rule['lower'], rule['upper']
                terms_map = rule['model_terms']
                
                for t in range(T):
                    vars_at_t = {c: model.x[c, t] for c in cols}
                    # 多项式：intercept + sum(coeff * term)
                    poly_expr = intercept + sum(
                        float(coeff) * self._parse_term(key, vars_at_t) 
                        for key, coeff in terms_map.items()
                    )
                    # 限制残差在指定范围内
                    residual = model.x[target, t] - poly_expr
                    model.row_cons.add(expr=inequality(l_bound, residual, u_bound))

        # 求解配置
        solver = SolverFactory('ipopt')
        # 设置求解时间限制或容差（可选）
        # solver.options['max_iter'] = 3000 
        
        results = solver.solve(model, tee=self.verbose)

        # 结果写回
        refined_df = data.copy()
        for n in cols:
            for t in range(T):
                # 只有在求解成功时提取值，否则保留原值（可选逻辑）
                try:
                    refined_df.iloc[t, cols.index(n)] = value(model.x[n, t])
                except ValueError:
                    continue
        
        return refined_df
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
    np.random.seed(42)
    T_size = 60
    t = np.linspace(0, 10, T_size)

    # col_1: 温度 (平滑的正弦波 + 噪声)
    col_1_true = 20 * np.sin(0.5 * t) + 50
    col_1_noisy = col_1_true + np.random.normal(0, 2, T_size)

    # col_3: 压力 (线性增长 + 噪声)
    col_3_true = 2 * t + 10
    col_3_noisy = col_3_true + np.random.normal(0, 0.5, T_size)

    # col_2: 产率 (物理公式: col_2 = -5 + 0.1*col_1 + 0.005*col_1^2 + 0.5*col_3)
    # 这是一个典型的非线性多项式关系
    col_2_true = -5 + 0.1 * col_1_true + 0.005 * (col_1_true**2) + 0.5 * col_3_true
    col_2_noisy = col_2_true + np.random.normal(0, 3, T_size) # 故意加入剧烈噪声

    raw_df = pd.DataFrame({'col_1': col_1_noisy, 'col_2': col_2_noisy, 'col_3': col_3_noisy})


    # --- 2. 准备你的约束格式 ---
    my_constraints = {
            'row_constraints': [{
            'target': 'col_2',
            'intercept': -5.0,
            'model_terms': {
                'col_1': 0.1,
                'col_1^2': 0.005,
                'col_3': 0.5
            },
            'lower': -0.5, # 强制要求修复后的数据必须极其贴合物理公式
            'upper': 0.5
        }],
            'speed_constraints': {'col_1': (-3.0, 3.0), 'col_2': (-5.0, 5.0), 'col_3': (-1.0, 1.0)},
            'accel_constraints': {'col_1': (-1.0, 1.0), 'col_2': (-2.0, 2.0), 'col_3': (-0.5, 0.5)}
        }

    # --- 3. 调用刚才封装的方法 ---
    # print("--- 原始数据 ---")
    # print(raw_df)

    # 这里调用我们之前定义的封装函数
    cleaned_df = _clean_with_mtclean(raw_df, my_constraints)

    # print("\n--- 清洗后的数据 ---")
    # print(cleaned_df.round(2))

    # --- 4. 验证效果 ---
    print("原始数据前5行：")
    print(raw_df.head())
    print("\n清洗后数据前5行：")
    print(cleaned_df.head())
