import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple


class SCREEN:
    """
    SCREEN algorithm for repairing multi-variate time series with speed constraints.
    
    Each variable has its own speed bounds in the form (min_speed, max_speed).
    Input: pd.DataFrame with 'timestamp' and variable columns.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 data_columns: List[str],
                 speed_constraints: Dict[str, Tuple[float, float]],
                 window_size: float):
        """
        Args:
            df: DataFrame with 'timestamp' and data columns
            data_columns: list of column names to repair, e.g., ['lon', 'lat']
            speed_constraints: dict like {'lon': (-0.8, 0.8), 'lat': (-0.5, 0.5)}
                               each value is (min_speed, max_speed)
            window_size: T, in timestamp units (float or int)
        """
        self.df = df.copy()
        self.data_columns = data_columns
        self.window_size = float(window_size)
        
        # 解析 speed_constraints 为 SMIN 和 SMAX 字典
        self.s_min = {}
        self.s_max = {}
        for col in self.data_columns:
            if col not in speed_constraints:
                raise ValueError(f"Missing speed constraint for column '{col}' in speed_constraints")
            bound = speed_constraints[col]
            if not (isinstance(bound, (list, tuple)) and len(bound) == 2):
                raise ValueError(f"speed_constraints['{col}'] must be tuple/list of (min, max)")
            s_min_val, s_max_val = bound
            if s_min_val > s_max_val:
                raise ValueError(f"In speed_constraints['{col}']: min ({s_min_val}) > max ({s_max_val})")
            self.s_min[col] = float(s_min_val)
            self.s_max[col] = float(s_max_val)
        
        # 添加 modify 列和 status 列
        for col in self.data_columns:
            self.df[f'{col}_modify'] = self.df[col].astype(float)
            self.df[f'{col}_status'] = 0  # 0: not fixed, 1: fixed
    
    def _get_local_candidates(self, 
                            window_df: pd.DataFrame, 
                            kp_row: pd.Series, 
                            pre_row: pd.Series,
                            var: str) -> Tuple[float, float, List[float]]:
        """
        Generate candidate values for key point kp based on speed bounds.
        
        Args:
            window_df: points in current sliding window
            kp_row: key point (first in window) to repair
            pre_row: last repaired point (predecessor)
            var: variable name
        
        Returns:
            lower_bound, upper_bound, candidates_list
        """
        pre_val = pre_row[f'{var}_modify']
        pre_time = pre_row['timestamp']
        kp_time = kp_row['timestamp']
        dt = kp_time - pre_time
        
        if abs(dt) < 1e-9:
            lower_bound = upper_bound = pre_val
        else:
            lower_bound = pre_val + self.s_min[var] * dt
            upper_bound = pre_val + self.s_max[var] * dt
        
        candidates = [kp_row[f'{var}_modify']]  # 原始值

        # 从窗口中其他点反推候选值
        for _, tp in window_df.iterrows():
            if abs(tp['timestamp'] - kp_time) < 1e-9:
                continue
            dt_kp = kp_time - tp['timestamp']
            val_min = tp[f'{var}_modify'] + self.s_min[var] * dt_kp
            val_max = tp[f'{var}_modify'] + self.s_max[var] * dt_kp
            candidates.append(val_min)
            candidates.append(val_max)
        
        return lower_bound, upper_bound, candidates
    
    def main_screen(self) -> pd.DataFrame:
        """
        Main SCREEN algorithm with sliding window.
        Returns: repaired DataFrame with `_modify` columns.
        """
        df = self.df.sort_values('timestamp').reset_index(drop=True)
        n = len(df)
        
        if n == 0:
            return self.df
        
        temp_df = pd.DataFrame()
        read_idx = 0
        
        # 初始化第一个点
        first_row = df.iloc[0]
        temp_df = pd.concat([temp_df, first_row.to_frame().T], ignore_index=True)
        w_start_time = first_row['timestamp']
        w_goal_time = w_start_time + self.window_size
        pre_point = first_row  # 上一个已修复的点

        read_idx = 1
        while read_idx < n:
            curr_row = df.iloc[read_idx]
            curr_time = curr_row['timestamp']
            
            if curr_time > w_goal_time:
                # 当前点超出当前窗口目标时间，开始滑动窗口并修复 kp
                while True:
                    if temp_df.empty:
                        # 窗口为空，直接加入当前点作为新窗口起点
                        temp_df = pd.concat([temp_df, curr_row.to_frame().T], ignore_index=True)
                        w_goal_time = curr_time + self.window_size
                        break
                    
                    # 取出窗口第一个点作为待修复点 (kp)
                    kp_row = temp_df.iloc[0]
                    w_start_time = kp_row['timestamp']
                    w_goal_time = w_start_time + self.window_size
                    
                    if curr_time <= w_goal_time:
                        # 当前点仍在窗口目标时间内，加入窗口
                        temp_df = pd.concat([temp_df, curr_row.to_frame().T], ignore_index=True)
                        break
                    
                    # 否则：修复 kp
                    self._local_repair(temp_df, pre_point)
                    pre_point = kp_row  # 更新上一个修复点
                    temp_df = temp_df.drop(temp_df.index[0]).reset_index(drop=True)
            else:
                # 当前点在窗口时间内，加入
                temp_df = pd.concat([temp_df, curr_row.to_frame().T], ignore_index=True)
            
            read_idx += 1
        
        # 处理剩余窗口中的点
        while len(temp_df) > 0:
            kp_row = temp_df.iloc[0]
            self._local_repair(temp_df, pre_point)
            pre_point = kp_row
            temp_df = temp_df.drop(temp_df.index[0]).reset_index(drop=True)
        
        return self.df

    def _local_repair(self, window_df: pd.DataFrame, pre_row: pd.Series):
        """Repair the first point (kp) in window_df using median of candidates."""
        kp_row = window_df.iloc[0]
        kp_time = kp_row['timestamp']
        
        for var in self.data_columns:
            lb, ub, candidates = self._get_local_candidates(window_df, kp_row, pre_row, var)
            candidates.sort()
            mid_idx = len(candidates) // 2
            x_mid = candidates[mid_idx]
            
            if x_mid > ub:
                repaired = ub
            elif x_mid < lb:
                repaired = lb
            else:
                repaired = x_mid
            
            # 更新全局 df
            idx = self.df[self.df['timestamp'] == kp_time].index[0]
            self.df.loc[idx, f'{var}_modify'] = repaired
            self.df.loc[idx, f'{var}_status'] = 1


if __name__ == "__main__":
    # 创建测试数据
    df = pd.DataFrame({
        'timestamp': [0, 1, 3, 5, 7, 10],
        'lon': [0.0, 0.5, 0.6, 1.0, 1.2, 2.0],
        'lat': [0.0, 2.0, 0.4, 0.8, 0.9, 1.0],   # t=1 异常
        'speed': [0.0, 50.0, 5.0, 10.0, 8.0, 12.0]  # t=1 异常
    })
    
    # 定义统一的速度约束字典
    speed_constraints = {
        'lon': (-0.8, 0.8),
        'lat': (-0.6, 0.6),
        'speed': (0.0, 20.0)  # 速度非负
    }
    
    # 执行 SCREEN
    screen = SCREEN(
        df=df,
        data_columns=['lon', 'lat', 'speed'],
        speed_constraints=speed_constraints,
        window_size=5
    )
    
    result = screen.main_screen()
    
    # 输出结果
    print(result[[
        'timestamp', 
        'lon', 'lon_modify', 
        'lat', 'lat_modify', 
        'speed', 'speed_modify'
    ]])