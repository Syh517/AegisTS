import numpy as np
import pandas as pd
from typing import Dict, Any, List

class SCREEN:
    def __init__(self, df: pd.DataFrame, data_columns: List[str], 
                 speed_constraints: Dict[str, Any], window_size: int):
        """
        :param df: 输入的 DataFrame
        :param data_columns: 需要清洗的列名列表
        :param speed_constraints: 速度约束字典 {'col_1': (min, max), ...}
        :param window_size: 滑动窗口大小 T
        """
        self.df = df.copy()
        self.data_columns = data_columns
        self.constraints = speed_constraints
        self.T = window_size
        # 自动识别时间列
        self.time_col = 'timestamp' if 'timestamp' in self.df.columns else self.df.columns[0]

    def main_screen(self) -> pd.DataFrame:
        """执行清洗并返回包含 {col}_modify 列的 DataFrame"""
        for col_name in self.data_columns:
            # 获取该列对应的约束，若无约束则跳过
            constraint = self.constraints.get(col_name)
            if constraint is None:
                # 如果没有约束，默认不修改，直接复制一列作为 modify
                self.df[f"{col_name}_modify"] = self.df[col_name]
                continue
            
            s_min, s_max = constraint
            self._repair_column(col_name, s_min, s_max)
            
        return self.df

    def _repair_column(self, col_name: str, s_min: float, s_max: float):
        timestamps = self.df[self.time_col].values
        values = self.df[col_name].values.astype(float)
        size = len(values)
        
        # 创建一个用于存储修改后值的数组
        modify_values = values.copy()
        
        buffer_indices = [0]
        w_start_time = timestamps[0]
        w_goal_time = w_start_time + self.T
        
        pre_point_idx = 0
        read_index = 1
        
        while read_index < size:
            cur_time = timestamps[read_index]
            
            if cur_time > w_goal_time:
                while True:
                    if not buffer_indices:
                        buffer_indices.append(read_index)
                        w_goal_time = cur_time + self.T
                        break
                    
                    kp_idx = buffer_indices[0]
                    w_start_time = timestamps[kp_idx]
                    w_goal_time = w_start_time + self.T
                    
                    if cur_time <= w_goal_time:
                        buffer_indices.append(read_index)
                        break
                    
                    self._local_repair(buffer_indices, modify_values, timestamps, 
                                     pre_point_idx, s_min, s_max)
                    
                    pre_point_idx = kp_idx
                    buffer_indices.pop(0)
            else:
                buffer_indices.append(read_index)
            
            read_index += 1
            
        # 补齐最后窗口的修复
        while buffer_indices:
            kp_idx = buffer_indices[0]
            self._local_repair(buffer_indices, modify_values, timestamps, 
                             pre_point_idx, s_min, s_max)
            pre_point_idx = kp_idx
            buffer_indices.pop(0)
            
        # 按照调用函数的要求，结果存入 {col}_modify 列
        self.df[f"{col_name}_modify"] = modify_values

    def _local_repair(self, buffer_indices, modify_values, timestamps, pre_idx, s_min, s_max):
        kp_idx = buffer_indices[0]
        kp_time = timestamps[kp_idx]
        pre_time = timestamps[pre_idx]
        pre_val = modify_values[pre_idx]
        
        # 计算基于前一个固定点的上下界
        lower_bound = pre_val + s_min * (kp_time - pre_time)
        upper_bound = pre_val + s_max * (kp_time - pre_time)
        
        # 收集候选值
        candidates = [modify_values[kp_idx]]
        for i in range(1, len(buffer_indices)):
            next_idx = buffer_indices[i]
            val = modify_values[next_idx]
            d_time = kp_time - timestamps[next_idx]
            candidates.append(val + s_min * d_time)
            candidates.append(val + s_max * d_time)
        
        # 中位数启发式修复
        x_mid = np.median(candidates)
        
        if x_mid > upper_bound:
            modify_values[kp_idx] = upper_bound
        elif x_mid < lower_bound:
            modify_values[kp_idx] = lower_bound
        else:
            modify_values[kp_idx] = x_mid