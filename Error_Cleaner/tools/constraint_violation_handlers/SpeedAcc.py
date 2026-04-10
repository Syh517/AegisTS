import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class SpeedAcc:
    def __init__(self, df: pd.DataFrame, time_col: str, data_cols: List[str], 
                 T: float, speed_constraints: Dict[str, Tuple[float, float]], 
                 acc_constraints: Dict[str, Tuple[float, float]]):
        """
        SpeedAcc 构造函数
        :param df: 输入的 DataFrame
        :param time_col: 时间列名
        :param data_cols: 需要清洗的列名列表
        :param T: 时间窗口大小
        :param speed_constraints: 速度约束字典 {'col': (min, max)}
        :param acc_constraints: 加速度约束字典 {'col': (min, max)}
        """
        self.df = df.copy()
        self.time_col = time_col
        self.data_cols = data_cols
        self.T = float(T)
        self.s_constraints = speed_constraints
        self.a_constraints = acc_constraints

    def run(self) -> pd.DataFrame:
        """执行清洗流程，返回带有 {col}_clean 列的 DataFrame"""
        for col in self.data_cols:
            s_bound = self.s_constraints.get(col)
            a_bound = self.a_constraints.get(col)
            
            # 只有当列同时拥有速度和加速度约束时才执行修复，否则原样输出
            if s_bound and a_bound:
                self._repair_column(col, s_bound, a_bound)
            else:
                self.df[f"{col}_clean"] = self.df[col]
        return self.df

    def _repair_column(self, col: str, s_bound: Tuple[float, float], a_bound: Tuple[float, float]):
        ts = self.df[self.time_col].values.astype(float)
        vals = self.df[col].values.astype(float)
        n = len(vals)
        modify = vals.copy()
        
        s_min, s_max = s_bound
        a_min, a_max = a_bound

        # 1. 初始点修复 (前两个点)
        modify[0] = self._get_median_candidate(0, ts, vals, s_min, s_max)
        
        if n > 1:
            lb_s, ub_s = self._calc_speed_bounds(0, 1, ts, modify, s_min, s_max)
            mid_val = self._get_median_candidate(1, ts, vals, s_min, s_max)
            modify[1] = np.clip(mid_val, lb_s, ub_s)

        # 2. 迭代修复 (从第三个点开始应用加速度约束)
        for k in range(2, n):
            # 速度边界
            lb_s, ub_s = self._calc_speed_bounds(k-1, k, ts, modify, s_min, s_max)
            # 加速度边界
            lb_a, ub_a = self._calc_accel_bounds(k-2, k-1, k, ts, modify, a_min, a_max)
            
            # 综合约束边界 (取交集)
            lower_bound = max(lb_s, lb_a)
            upper_bound = min(ub_s, ub_a)

            # 获取窗口内候选值的点
            w_end = k + 1
            while w_end < n and ts[w_end] - ts[k] <= self.T:
                w_end += 1
            
            candidates = [vals[k]]
            for j in range(k + 1, w_end):
                dt_kj = ts[k] - ts[j]
                candidates.append(vals[j] + s_min * dt_kj)
                candidates.append(vals[j] + s_max * dt_kj)
            
            x_mid = np.median(candidates)
            modify[k] = np.clip(x_mid, lower_bound, upper_bound)

        self.df[f"{col}_clean"] = modify

    def _calc_speed_bounds(self, i, j, ts, modify, s_min, s_max):
        dt = ts[j] - ts[i]
        return modify[i] + s_min * dt, modify[i] + s_max * dt

    def _calc_accel_bounds(self, k2, k1, k, ts, modify, a_min, a_max):
        dt1 = ts[k1] - ts[k2]
        dt2 = ts[k] - ts[k1]
        # 计算前一点的速度 v = dx/dt
        v_pre = (modify[k1] - modify[k2]) / dt1 if dt1 > 1e-9 else 0
        # 根据加速度公式：x_k = x_k-1 + v_pre*dt2 + 0.5*a*dt2^2 (Java原版简化为 v*dt 线性叠加)
        # 这里遵循 Java 源码中的 v 线性预测逻辑
        lb = modify[k1] + (v_pre + a_min * dt2) * dt2
        ub = modify[k1] + (v_pre + a_max * dt2) * dt2
        return lb, ub

    def _get_median_candidate(self, k, ts, vals, s_min, s_max):
        """窗口内候选值中位数"""
        w_end = k + 1
        while w_end < len(ts) and ts[w_end] - ts[k] <= self.T:
            w_end += 1
        candidates = [vals[k]]
        for j in range(k + 1, w_end):
            dt = ts[k] - ts[j]
            candidates.extend([vals[j] + s_min * dt, vals[j] + s_max * dt])
        return np.median(candidates)