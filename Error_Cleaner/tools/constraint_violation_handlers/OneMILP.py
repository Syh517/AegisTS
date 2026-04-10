import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

class OneMILP:
    def __init__(self, timestamps, df_obs, constraints, T_window, M=10000):
        self.timestamps = timestamps
        self.df_obs = df_obs
        self.constraints = constraints
        self.T_window = T_window
        self.M = M
        self.column_names = list(df_obs.columns)
        self.N = len(self.column_names)

    def clean_data_segmented(self, segment_size=100, overlap=10):
        """
        :param segment_size: 每段处理的数据点数 (建议 50-150 以适应限制)
        :param overlap: 段与段之间的重叠点数，用于保证边缘平滑
        """
        L_total = len(self.timestamps)
        cleaned_values = np.zeros((L_total, self.N))
        z_values = np.zeros(L_total)
        
        # 记录每个点是否已被填充（处理最后一段可能不整除的情况）
        filled_mask = np.zeros(L_total, dtype=bool)

        start_idx = 0
        while start_idx < L_total:
            end_idx = min(start_idx + segment_size, L_total)
            
            # 提取当前分段数据
            curr_ts = self.timestamps[start_idx:end_idx]
            curr_obs = self.df_obs.iloc[start_idx:end_idx]
            
            print(f"正在求解分段: Index {start_idx} 到 {end_idx}...")
            
            # 调用核心 MILP 求解单段
            seg_cleaned, seg_z = self._solve_single_segment(curr_ts, curr_obs)
            
            if seg_cleaned is not None:
                # 填充结果：如果是第一段，全部填充；
                # 如果不是第一段，为了边缘平滑，重叠部分可以只取当前段的结果
                fill_start = start_idx if start_idx == 0 else start_idx + overlap // 2
                
                # 对应到当前 seg_cleaned 中的索引
                seg_offset = fill_start - start_idx
                
                actual_fill_end = end_idx
                # 如果不是最后一段，则只填充到 end_idx - overlap // 2
                if end_idx < L_total:
                    actual_fill_end = end_idx - overlap // 2
                
                fill_len = actual_fill_end - fill_start
                
                cleaned_values[fill_start:actual_fill_end] = seg_cleaned[seg_offset:seg_offset+fill_len]
                z_values[fill_start:actual_fill_end] = seg_z[seg_offset:seg_offset+fill_len]
            else:
                print(f"警告: 分段 {start_idx}-{end_idx} 求解失败。")

            # 更新起始位置：移动 segment_size - overlap
            if end_idx == L_total:
                break
            start_idx += (segment_size - overlap)

        return cleaned_values, z_values

    def _solve_single_segment(self, timestamps, df_obs):
        L = len(timestamps)
        obs_vals = df_obs.values
        
        try:
            model = gp.Model("Segment_Cleaning")
            model.setParam('OutputFlag', 1) # 打开日志看看到底是多少个约束
            
            x = model.addVars(L, self.N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
            z = model.addVars(L, vtype=GRB.BINARY, name="z")

            model.setObjective(gp.quicksum(z[i] for i in range(L)), GRB.MINIMIZE)

            # --- 修改处：精简约束逻辑 ---
            for i in range(L - 1):
                # 仅约束紧邻的下一个点，这能极大减少约束数量
                j = i + 1 
                delta_t = timestamps[j] - timestamps[i]
                
                # 如果相邻点间隔太大超过了窗口，可以选择跳过或只约束步长
                if delta_t <= self.T_window:
                    for d, col_name in enumerate(self.column_names):
                        s_min, s_max = self.constraints[col_name]
                        model.addConstr(x[j, d] - x[i, d] <= s_max * delta_t)
                        model.addConstr(x[j, d] - x[i, d] >= s_min * delta_t)
            # --------------------------

            # Big-M 约束保持
            for i in range(L):
                for d in range(self.N):
                    model.addConstr(x[i, d] - self.M * z[i] <= obs_vals[i, d])
                    model.addConstr(x[i, d] + self.M * z[i] >= obs_vals[i, d])

            model.optimize()

            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                res_x = np.array([[x[i, d].X for d in range(self.N)] for i in range(L)])
                res_z = [z[i].X for i in range(L)]
                return res_x, res_z
            return None, None

        except gp.GurobiError:
            return None, None

# 使用示例
# milp = SegmentedMILP(timestamps, df_obs, constraints, T_window=60)
# cleaned_data, outliers = milp.clean_data_segmented(segment_size=80, overlap=10)