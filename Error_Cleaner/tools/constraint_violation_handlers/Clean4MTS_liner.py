import numpy as np
import pandas as pd
import pygad
from typing import Dict, List, Any, Tuple

# --- 1. 行约束适配器 (保持不变) ---
class LinearRuleModel:
    def __init__(self, rule_dict: Dict):
        self.target = rule_dict['target']
        self.features = rule_dict['features']
        self.intercept = rule_dict['intercept']
        self.coefs = rule_dict['model_terms']
        self.lower = rule_dict['lower']
        self.upper = rule_dict['upper']
        self.w = np.array([self.coefs.get(f, 0.0) for f in self.features])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.intercept

# --- 2. 核心清洗类 (PyGAD 版) ---
class Clean4MTS:
    def __init__(self, dataframe: pd.DataFrame, speed_constraint: Dict, 
                 acceleration_constraints: Dict, row_constraints: List[Dict],
                 beta=1.2, w_modification=0.05):
        self.df = dataframe.copy().astype(np.float64)
        self.var_names = list(speed_constraint.keys())
        self.sc = speed_constraint
        self.ac = acceleration_constraints
        self.rules = [LinearRuleModel(r) for r in row_constraints]
        self.beta = beta
        self.w_mod = w_modification
        self.vtw_index = []

    def _get_physical_bounds(self, idx: int) -> List[Dict[str, float]]:
        """计算 PyGAD 所需的基因范围参数"""
        dt = 1.0
        gene_space = []
        
        for var in self.var_names:
            v_min, v_max = self.sc[var]
            val_t_1 = self.df.loc[idx - 1, var]
            
            # 速度约束范围
            s_lb = val_t_1 + v_min * dt
            s_ub = val_t_1 + v_max * dt
            
            # 融合加速度约束
            if idx >= 2:
                v_prev = (val_t_1 - self.df.loc[idx - 2, var]) / dt
                a_min, a_max = self.ac.get(var, (-np.inf, np.inf))
                a_lb = val_t_1 + v_prev * dt + a_min * (dt**2)
                a_ub = val_t_1 + v_prev * dt + a_max * (dt**2)
                final_lb, final_ub = max(s_lb, a_lb), min(s_ub, a_ub)
                if final_lb > final_ub: final_lb, final_ub = s_lb, s_ub
            else:
                final_lb, final_ub = s_lb, s_ub
            
            gene_space.append({'low': final_lb, 'high': final_ub})
        return gene_space

    def violation_profiling(self):
        """异常检测逻辑"""
    def violation_profiling(self):
        """增强后的异常检测，同时考虑行约束、速度和加速度"""
        errors = np.zeros(len(self.df), dtype=bool)
        
        # 1. 检查行约束 (Row Constraints)
        for rule in self.rules:
            res = self.df[rule.target] - rule.predict(self.df[rule.features].values)
            err = (res > rule.upper) | (res < rule.lower)
            errors = errors | err.values
            
        # 2. 检查速度约束 (Speed Constraints)
        dt = 1.0
        for var, (v_min, v_max) in self.sc.items():
            diff = self.df[var].diff() / dt
            # 排除 NaN (第一行)
            err_speed = (diff > v_max) | (diff < v_min)
            errors = errors | err_speed.fillna(False).values

        # 3. 提取索引并修复
        idx = np.where(errors)[0]
        idx = idx[(idx > 1) & (idx < len(self.df) - 1)]
        if len(idx) == 0: return
        
        start_node = idx[0]
        for i in range(1, len(idx)):
            if idx[i] > idx[i-1] + 1:
                self.vtw_index.append((start_node, idx[i-1]))
                start_node = idx[i]
        self.vtw_index.append((start_node, idx[-1]))

    def repair(self):
        """使用 PyGAD 进行并行修复"""
        for s, e in self.vtw_index:
            for i in range(s, e + 1):
                gene_space = self._get_physical_bounds(i)
                original_val = self.df.loc[i, self.var_names].values

                # 定义适应度函数 (PyGAD 要求越大越好，所以取惩罚值的负数)
                def fitness_func(ga_instance, solution, solution_idx):
                    repaired_dict = dict(zip(self.var_names, solution))
                    
                    # 1. 计算行约束违反代价
                    penalty = 0
                    for rule in self.rules:
                        X = np.array([repaired_dict[f] for f in rule.features])
                        Y = repaired_dict[rule.target]
                        residual = Y - rule.predict(X)
                        deg = max(0, max(residual - rule.upper, rule.lower - residual))
                        norm = (rule.upper - rule.lower) / 2.0 + 1e-6
                        penalty += (1 - np.exp(-deg / (norm * self.beta)))
                    
                    # 2. 计算最小修改代价 (L2)
                    mod_penalty = self.w_mod * np.mean(np.square(solution - original_val))
                    
                    return -(penalty + mod_penalty)

                # 配置 PyGAD
                ga_instance = pygad.GA(
                    num_generations=50,
                    num_parents_mating=5,
                    fitness_func=fitness_func,
                    sol_per_pop=20,
                    num_genes=len(self.var_names),
                    gene_space=gene_space,
                    parent_selection_type="sss", # 稳态选择
                    crossover_type="single_point",
                    mutation_type="random",
                    mutation_probability=0.1,
                    suppress_warnings=True
                )

                ga_instance.run()
                solution, solution_fitness, _ = ga_instance.best_solution()
                self.df.loc[i, self.var_names] = solution

    def data_cleaning(self) -> pd.DataFrame:
        self.violation_profiling()
        self.repair()
        return self.df

# --- 3. 最终接口适配 ---
def _clean_with_clean4mts(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    try:
        # 安装提示：如果环境中没有 pygad，请执行 pip install pygad
        cleaner = Clean4MTS(
            dataframe=data,
            speed_constraint=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            row_constraints=constraints.get('row_constraints', [])
        )
        return cleaner.data_cleaning()
    except Exception as e:
        print(f"Clean4MTS (PyGAD) 运行出错: {e}")
        return data


if __name__ == '__main__':


    # 1. 生成正常基础数据
    t = np.linspace(0, 10, 100)
    col_3 = np.sin(t) * 10
    col_5 = np.cos(t) * 5
    # 根据公式：col_1 = 0.12 + 1.0*col_3 + 0.99*col_5
    col_1 = 0.12 + 1.0 * col_3 + 0.99 * col_5

    df = pd.DataFrame({'col_1': col_1, 'col_3': col_3, 'col_5': col_5})

    # 2. 注入异常
    # 异常点 A: 速度突跳 (col_1 在第10行突然暴增，违反速度约束)
    df.loc[10, 'col_1'] += 20 

    # 异常点 B: 逻辑偏差 (col_3 在第50行发生漂移，虽然数值平滑，但违反了 col_1/3/5 的比例关系)
    df.loc[50, 'col_3'] += 5 

    # 3. 定义约束字典 (即你提供的格式)
    constraints = {
        'speed_constraints': {
            'col_1': (-2.0, 2.0), 
            'col_3': (-3.0, 3.0), 
            'col_5': (-2.0, 2.0)
        },
        'accel_constraints': {
            'col_1': (-1.0, 1.0),
            'col_3': (-1.0, 1.0),
            'col_5': (-1.0, 1.0)
        },
        'row_constraints': [
            {
                'target': 'col_1',
                'features': ['col_3', 'col_5'],
                'model_terms': {'col_3': 1.0000, 'col_5': 0.9922},
                'intercept': 0.1273,
                'lower': -0.6368,
                'upper': 0.6607
            }
        ]
    }

    # 调用封装好的函数
    cleaned_df = _clean_with_clean4mts(df, constraints)

    # 查看修复效果
    print("原始异常点数据 (第10行 col_1):", df.loc[10, 'col_1'])
    print("修复后点数据 (第10行 col_1):", cleaned_df.loc[10, 'col_1'])

    print("\n原始异常点数据 (第50行 col_3):", df.loc[50, 'col_3'])
    print("修复后点数据 (第50行 col_3):", cleaned_df.loc[50, 'col_3'])