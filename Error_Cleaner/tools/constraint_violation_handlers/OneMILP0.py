import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Tuple

# class OneMILP:
#     def __init__(self, speed_constraints: Dict[str, Tuple[float, float]], T: int = 5):
#         self.speed_constraints = speed_constraints
#         self.T = T

#     def mainGlobal(self, df: pd.DataFrame) -> pd.DataFrame:
#         df = df.sort_values('timestamp').reset_index(drop=True).copy()
#         n = len(df)
#         if n < 2:
#             return df

#         timestamps = df["timestamp"].values.astype(float)
#         all_cols = list(self.speed_constraints.keys())
#         org_vals = {col: df[col].values.astype(float) for col in all_cols}

#         repaired_vals, is_fixed_array = self._build_joint_milp(org_vals, timestamps, all_cols)

#         for col in all_cols:
#             df[f"{col}_repaired"] = repaired_vals[col]
#         df["is_fixed"] = is_fixed_array
#         return df

#     def _build_joint_milp(self, org_vals, timestamps, columns):
#         n = len(timestamps)
#         model = gp.Model("Optimized_MILP")
#         model.setParam("OutputFlag", 0)
#         # 提高数值稳定性：设置可行性精度
#         model.setParam("FeasibilityTol", 1e-6)

#         # 1. 变量定义
#         x_vars = {}
#         for col in columns:
#             # 根据原始数据的波动范围确定界限，避免无限空间的搜索
#             v_min, v_max = org_vals[col].min(), org_vals[col].max()
#             margin = (v_max - v_min) * 2 + 10
#             x_vars[col] = model.addVars(n, lb=v_min - margin, ub=v_max + margin, name=f"x_{col}")
        
#         z_vars = model.addVars(n, vtype=GRB.BINARY, name="z")

#         # 2. 目标函数优化：最小化修改点数 + 微弱的原始偏差惩罚（保证修复值尽量靠近原值）
#         # 这可以防止修复值“乱跑”
#         soft_penalty = 0
#         for col in columns:
#             for i in range(n):
#                 diff = model.addVar(lb=0, name=f"diff_{col}_{i}")
#                 model.addConstr(diff >= x_vars[col][i] - org_vals[col][i])
#                 model.addConstr(diff >= org_vals[col][i] - x_vars[col][i])
#                 soft_penalty += 1e-5 * diff

#         model.setObjective(gp.quicksum(z_vars[i] for i in range(n)) + soft_penalty, GRB.MINIMIZE)

#         # 3. 速度约束优化
#         for col in columns:
#             smin, smax = self.speed_constraints[col]
#             for i in range(n):
#                 # 效率优化：只约束时间窗口 T 内的点，且限制最大比较对数
#                 for j in range(i + 1, min(i + 5, n)): # 增加邻近搜索限制
#                     dt = timestamps[j] - timestamps[i]
#                     if dt <= 0: continue
#                     if dt > self.T: break
                    
#                     model.addConstr(x_vars[col][j] - x_vars[col][i] <= smax * dt)
#                     model.addConstr(x_vars[col][j] - x_vars[col][i] >= smin * dt)

#         # 4. 动态 Big-M 约束
#         for col in columns:
#             # M 的大小至少应覆盖该列的最大可能跳变
#             M_val = (org_vals[col].max() - org_vals[col].min()) * 10 + 100
#             for i in range(n):
#                 model.addConstr(x_vars[col][i] <= org_vals[col][i] + M_val * z_vars[i])
#                 model.addConstr(x_vars[col][i] >= org_vals[col][i] - M_val * z_vars[i])

#         model.optimize()

#         if model.status != GRB.OPTIMAL:
#             return org_vals, np.zeros(n, dtype=bool)

#         # 5. 提取结果
#         repaired = {col: np.array([x_vars[col][i].X for i in range(n)]) for col in columns}
#         # 判定是否修改：基于二进制变量
#         is_fixed = np.array([z_vars[i].X > 0.5 for i in range(n)])

#         return repaired, is_fixed

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Tuple, List

class OneMILP:
    def __init__(self, speed_constraints: Dict[str, Tuple[float, float]], 
                 T: int = 5, batch_size: int = 50, overlap: int = 5):
        """
        :param speed_constraints: {col: (SMIN, SMAX)}
        :param T: 时间窗口约束范围
        :param batch_size: 每个 batch 的点数（建议 50-80 以避开 License 限制）
        :param overlap: 重叠点数（用于衔接参考）
        """
        self.speed_constraints = speed_constraints
        self.T = T
        self.batch_size = batch_size
        self.overlap = overlap

    def mainGlobal(self, df: pd.DataFrame) -> pd.DataFrame:
        """入口函数：负责分段逻辑"""
        df = df.sort_values('timestamp').reset_index(drop=True).copy()
        n = len(df)
        all_cols = list(self.speed_constraints.keys())

        if n <= self.batch_size:
            return self._repair_chunk(df, all_cols)

        repaired_chunks = []
        start_idx = 0
        last_fix_anchor = None  # 用于记录上一个 batch 的最后一个修复点

        while start_idx < n:
            end_idx = min(start_idx + self.batch_size, n)
            batch = df.iloc[start_idx:end_idx].copy()
            
            # 执行局部清洗
            repaired_batch = self._repair_chunk(batch, all_cols, anchor=last_fix_anchor)
            
            # 更新锚点：为下一个 batch 提供衔接参考
            last_fix_anchor = {
                'timestamp': repaired_batch['timestamp'].iloc[-1],
                'vals': {col: repaired_batch[f"{col}_repaired"].iloc[-1] for col in all_cols}
            }
            
            repaired_chunks.append(repaired_batch)
            
            # 步进：移动 batch_size - overlap
            start_idx += (self.batch_size - self.overlap)
            if end_idx == n: break

        # 合并结果，重叠部分以最新的修复为准
        full_df = pd.concat(repaired_chunks)
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        return full_df.sort_index()

    def _repair_chunk(self, chunk_df, columns, anchor=None):
        """核心 MILP 计算单元"""
        n = len(chunk_df)
        timestamps = chunk_df["timestamp"].values.astype(float)
        org_vals = {col: chunk_df[col].values.astype(float) for col in columns}

        model = gp.Model("Chunk_MILP")
        model.setParam("OutputFlag", 0)
        model.setParam("FeasibilityTol", 1e-6)

        # 1. 变量定义
        x_vars = {}
        for col in columns:
            v_min, v_max = org_vals[col].min(), org_vals[col].max()
            margin = (v_max - v_min) * 2 + 10
            x_vars[col] = model.addVars(n, lb=v_min - margin, ub=v_max + margin, name=f"x_{col}")
        
        z_vars = model.addVars(n, vtype=GRB.BINARY, name="z")

        # 2. 衔接约束：如果存在锚点，强制第一个点接上上一个 batch 的终点
        if anchor is not None:
            dt_anchor = timestamps[0] - anchor['timestamp']
            if dt_anchor > 0 and dt_anchor <= self.T:
                for col in columns:
                    smin, smax = self.speed_constraints[col]
                    model.addConstr(x_vars[col][0] - anchor['vals'][col] <= smax * dt_anchor)
                    model.addConstr(x_vars[col][0] - anchor['vals'][col] >= smin * dt_anchor)

        # 3. 目标函数：最小化 z + L1 偏差惩罚
        soft_penalty = 0
        for col in columns:
            for i in range(n):
                diff = model.addVar(lb=0)
                model.addConstr(diff >= x_vars[col][i] - org_vals[col][i])
                model.addConstr(diff >= org_vals[col][i] - x_vars[col][i])
                soft_penalty += 1e-5 * diff
        model.setObjective(gp.quicksum(z_vars[i] for i in range(n)) + soft_penalty, GRB.MINIMIZE)

        # 4. 速度约束
        for col in columns:
            smin, smax = self.speed_constraints[col]
            for i in range(n):
                for j in range(i + 1, min(i + 5, n)):
                    dt = timestamps[j] - timestamps[i]
                    if dt <= 0 or dt > self.T: break
                    model.addConstr(x_vars[col][j] - x_vars[col][i] <= smax * dt)
                    model.addConstr(x_vars[col][j] - x_vars[col][i] >= smin * dt)

        # 5. 动态 Big-M
        for col in columns:
            M_val = (org_vals[col].max() - org_vals[col].min()) * 5 + 100
            for i in range(n):
                model.addConstr(x_vars[col][i] <= org_vals[col][i] + M_val * z_vars[i])
                model.addConstr(x_vars[col][i] >= org_vals[col][i] - M_val * z_vars[i])

        model.optimize()

        # 提取结果并返回
        res_df = chunk_df.copy()
        if model.status == GRB.OPTIMAL:
            for col in columns:
                res_df[f"{col}_repaired"] = [x_vars[col][i].X for i in range(n)]
            res_df["is_fixed"] = [z_vars[i].X > 0.5 for i in range(n)]
        else:
            for col in columns: res_df[f"{col}_repaired"] = res_df[col]
            res_df["is_fixed"] = False
        
        return res_df