import pandas as pd
import numpy as np
import pygad
from typing import List, Dict, Any

# --- 1. 多项式规则模型 (支持非线性解析) ---
class PolyRuleModel:
    def __init__(self, rule_dict: Dict):
        self.target = rule_dict['target']
        self.intercept = rule_dict['intercept']
        self.lower = rule_dict['lower']
        self.upper = rule_dict['upper']
        self.model_terms = rule_dict['model_terms'] 

    def _parse_and_eval_term(self, term_key: str, data_map: Dict[str, float]) -> float:
        """解析项：支持 'col_1', 'col_1^2', 'col_3 col_1'"""
        # 处理交互项 (空格分隔)
        if ' ' in term_key:
            res = 1.0
            for part in term_key.split(' '):
                res *= self._parse_and_eval_term(part.strip(), data_map)
            return res
        
        # 处理高次项 (^)
        if '^' in term_key:
            base, expo = term_key.split('^')
            return data_map[base.strip()] ** float(expo)
        
        # 基础项
        return data_map[term_key.strip()]

    def predict_single(self, data_map: Dict[str, float]) -> float:
        """计算单点预测值"""
        y_hat = self.intercept
        for term, coeff in self.model_terms.items():
            y_hat += float(coeff) * self._parse_and_eval_term(term, data_map)
        return y_hat

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """向量化预测整个 DataFrame (用于异常检测)"""
        y_hat = np.full(len(df), self.intercept)
        for term, coeff in self.model_terms.items():
            if ' ' in term:
                parts = term.split(' ')
                val = df[parts[0].strip()].copy()
                for p in parts[1:]:
                    val *= df[p.strip()]
            elif '^' in term:
                base, expo = term.split('^')
                val = df[base.strip()] ** float(expo)
            else:
                val = df[term.strip()]
            y_hat += float(coeff) * val.values
        return y_hat

# --- 2. 核心清洗类 (PyGAD + 多项式版) ---
class Clean4MTS:
    def __init__(self, dataframe: pd.DataFrame, speed_constraint: Dict, 
                 acceleration_constraints: Dict, row_constraints: List[Dict],
                 beta=1.2, w_modification=0.05):
        self.df = dataframe.copy().astype(np.float64)
        self.var_names = list(speed_constraint.keys())
        self.sc = speed_constraint
        self.ac = acceleration_constraints
        self.rules = [PolyRuleModel(r) for r in row_constraints]
        self.beta = beta
        self.w_mod = w_modification
        self.vtw_index = []

    def _get_physical_bounds(self, idx: int) -> List[Dict[str, float]]:
        """计算基因搜索空间：融合速度与加速度约束"""
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
        """异常检测：行约束(多项式) + 速度 + 加速度"""
        errors = np.zeros(len(self.df), dtype=bool)
        
        # 1. 检查多项式行约束
        for rule in self.rules:
            res = self.df[rule.target] - rule.predict_df(self.df)
            err = (res > rule.upper) | (res < rule.lower)
            errors = errors | err.values
            
        # 2. 检查速度约束
        for var, (v_min, v_max) in self.sc.items():
            diff = self.df[var].diff()
            err_speed = (diff > v_max) | (diff < v_min)
            errors = errors | err_speed.fillna(False).values

        # 3. 检查加速度约束
        for var, (a_min, a_max) in self.ac.items():
            accel = self.df[var].diff().diff()
            err_accel = (accel > a_max) | (accel < a_min)
            errors = errors | err_accel.fillna(False).values

        # 4. 区间提取
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
        """GA 修复"""
        for s, e in self.vtw_index:
            for i in range(s, e + 1):
                gene_space = self._get_physical_bounds(i)
                original_val = self.df.loc[i, self.var_names].values

                def fitness_func(ga_instance, solution, solution_idx):
                    repaired_dict = dict(zip(self.var_names, solution))
                    penalty = 0
                    for rule in self.rules:
                        y_hat = rule.predict_single(repaired_dict)
                        residual = repaired_dict[rule.target] - y_hat
                        deg = max(0, residual - rule.upper, rule.lower - residual)
                        norm = (rule.upper - rule.lower) / 2.0 + 1e-6
                        penalty += (1 - np.exp(-deg / (norm * self.beta)))
                    
                    mod_penalty = self.w_mod * np.mean(np.square(solution - original_val))
                    return -(penalty + mod_penalty)

                ga_instance = pygad.GA(
                    num_generations=50, num_parents_mating=5, fitness_func=fitness_func,
                    sol_per_pop=20, num_genes=len(self.var_names), gene_space=gene_space,
                    parent_selection_type="sss", suppress_warnings=True
                )
                ga_instance.run()
                solution, _, _ = ga_instance.best_solution()
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
    data = pd.DataFrame({
        'col_1': [10.0, 10.5, 13.0, 11.0, 11.2], # col_1 在索引2处有一个跳变
        'col_2': [100.0, 101.0, 125.0, 102.0, 103.0], 
        'col_3': [5.0, 5.1, 5.2, 5.1, 5.2]
    })

    # 3. 定义约束字典 (即你提供的格式)
    constraints = {
        'speed_constraints': {'col_1': (-1.0, 1.0), 'col_2': (-5.0, 5.0), 'col_3': (-0.5, 0.5)},
        'accel_constraints': {'col_1': (-0.5, 0.5), 'col_2': (-2.0, 2.0), 'col_3': (-0.2, 0.2)},
        'row_constraints': [{
        'target': 'col_2',
        'model_terms': {'col_1^2': 0.5, 'col_3': 10.0},
        'intercept': -1.0,
        'lower': -0.5, 'upper': 0.5
        }]
    }

    # 调用封装好的函数
    cleaned_df = _clean_with_clean4mts(data, constraints)   
    # 查看修复效果
    print("原始异常点数据 (第2行 col_1):", data.loc[2, 'col_1'])
    print("修复后点数据 (第2行 col_1):", cleaned_df.loc[2, 'col_1'])
