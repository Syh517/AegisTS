import numpy as np
import pandas as pd
from typing import Dict, Any

class MTCSC_N:
    def __init__(self, df: pd.DataFrame, speed_constraints: Dict[str, Any], t: int):
        """
        :param df: 第一列为 timestamp，其余为数据列
        :param speed_constraints: 格式 {'col_1': (min, max), ...}
        :param t: 时间窗口跨度
        """
        self.original_df = df.copy()
        # 假设第一列是时间戳，其余是特征
        self.time_col = df.columns[0]
        self.feature_cols = df.columns[1:]
        
        self.timestamps = df[self.time_col].values.astype(np.int64)
        self.data = df[self.feature_cols].values.astype(np.float64)
        
        self.n_dims = len(self.feature_cols)
        self.window_t = t
        
        # 解析速度约束向量
        self.s_max_vec = np.zeros(self.n_dims)
        for i, col_name in enumerate(self.feature_cols):
            if col_name in speed_constraints:
                # 获取元组中的最大值作为该维度的 SMAX
                self.s_max_vec[i] = speed_constraints[col_name][1]
            else:
                self.s_max_vec[i] = np.inf

    def _is_speed_ok(self, val_a, val_b, time_diff):
        if time_diff <= 0: return True
        return np.all(np.abs(val_a - val_b) <= self.s_max_vec * time_diff)

    def _judge_modify(self, pre_val, max_val, kp_val, pre_t, max_t, kp_t):
        check_pre = self._is_speed_ok(pre_val, kp_val, kp_t - pre_t)
        check_max = self._is_speed_ok(max_val, kp_val, max_t - kp_t)
        return not (check_pre and check_max)

    def _local_repair(self, window_indices, pre_idx, kp_idx, cleaned_data):
        length = len(window_indices)
        pre_val, pre_t = cleaned_data[pre_idx], self.timestamps[pre_idx]
        kp_val, kp_t = cleaned_data[kp_idx], self.timestamps[kp_idx]

        if length == 1:
            if self._judge_modify(pre_val, pre_val, kp_val, pre_t, pre_t, kp_t):
                cleaned_data[kp_idx] = pre_val.copy()
            return

        top = np.zeros(length, dtype=int)
        lengths = np.zeros(length, dtype=int)
        top_index = -1

        for i in range(1, length):
            idx = window_indices[i]
            if self._is_speed_ok(pre_val, cleaned_data[idx], self.timestamps[idx] - pre_t):
                top[i], lengths[i], top_index = -1, 1, i
                break
        
        if top_index != -1:
            for i in range(top_index + 1, length):
                curr_idx, prev_idx = window_indices[i], window_indices[i-1]
                if self._is_speed_ok(cleaned_data[curr_idx], cleaned_data[prev_idx], 
                                   self.timestamps[curr_idx] - self.timestamps[prev_idx]):
                    if top[i-1] == -1: top[i], lengths[i-1] = i-1, lengths[i-1]+1
                    elif top[i-1] > 0: top[i], lengths[top[i-1]] = top[i-1], lengths[top[i-1]]+1
                else:
                    for j in range(i-1, top_index - 1, -1):
                        if self._is_speed_ok(cleaned_data[curr_idx], cleaned_data[window_indices[j]], 
                                            self.timestamps[curr_idx] - self.timestamps[window_indices[j]]):
                            if top[j] == -1: top[i], lengths[j] = j, lengths[j]+1
                            elif top[j] > 0: top[i], lengths[top[j]] = top[j], lengths[top[j]]+1
                            break
                        if j == top_index and self._is_speed_ok(pre_val, cleaned_data[curr_idx], self.timestamps[curr_idx] - pre_t):
                            top[i], lengths[i] = -1, 1

        max_idx_in_temp = max(top_index, 0)
        for i in range(max_idx_in_temp, length):
            if lengths[i] > lengths[max_idx_in_temp]: max_idx_in_temp = i

        max_point_idx = window_indices[max_idx_in_temp]
        max_val, max_t = cleaned_data[max_point_idx], self.timestamps[max_point_idx]

        if self._judge_modify(pre_val, max_val, kp_val, pre_t, max_t, kp_t):
            allowed_dis = self.s_max_vec * (kp_t - pre_t)
            total_dis = np.abs(max_val - pre_val)
            ratios = np.divide(allowed_dis, total_dis, out=np.ones_like(total_dis), where=total_dis!=0)
            ratio = min(np.min(ratios), 1.0)
            cleaned_data[kp_idx] = pre_val + (max_val - pre_val) * ratio

    def mainScreen(self):
        size = len(self.data)
        cleaned_values = self.data.copy()
        temp_indices = []
        pre_point_idx = 0
        pre_end = -1
        w_goal_time = self.timestamps[0] + self.window_t

        for read_index in range(size):
            cur_time = self.timestamps[read_index]
            if cur_time > w_goal_time:
                while temp_indices:
                    kp_idx = temp_indices[0]
                    w_goal_time = self.timestamps[kp_idx] + self.window_t
                    if cur_time <= w_goal_time:
                        temp_indices.append(read_index)
                        break
                    if pre_end == -1: pre_point_idx = kp_idx
                    self._local_repair(temp_indices, pre_point_idx, kp_idx, cleaned_values)
                    pre_point_idx, pre_end = kp_idx, 1
                    temp_indices.pop(0)
                if not temp_indices:
                    temp_indices.append(read_index)
                    w_goal_time = cur_time + self.window_t
            else:
                temp_indices.append(read_index)

        while temp_indices:
            kp_idx = temp_indices.pop(0)
            self._local_repair(temp_indices + [kp_idx], pre_point_idx, kp_idx, cleaned_values)
            pre_point_idx = kp_idx

        # 构建返回结果集，满足调用函数对 col_modify 的提取要求
        result_df = pd.DataFrame({self.time_col: self.timestamps})
        for i, col in enumerate(self.feature_cols):
            result_df[f"{col}_modify"] = cleaned_values[:, i]
        return result_df

# --- 调用函数保持不变 ---
def _clean_with_mtcsc_N(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用MTCSC_N清洗数据"""
    try:
        cleaner = MTCSC_N(
            df=data,
            speed_constraints=constraints.get('speed_constraints', {}),
            t=5
        )
        result = cleaner.mainScreen()
        cleaned_data = data.copy()
        for col in data.columns:
            modify_col = f"{col}_modify"
            if modify_col in result.columns:
                cleaned_data[col] = result[modify_col].values
        return cleaned_data
    except Exception as e:
        print(f"使用MTCSC_N清洗数据时出错: {e}")
        return data
    


import pandas as pd
import numpy as np
from typing import Dict, Any

# 1. 模拟生成一份带有噪声的多变量时序数据
def generate_sample_data():
    np.random.seed(42)
    n_points = 10
    
    # 第一列：时间戳 (假设每秒一个点)
    timestamps = np.arange(1000, 1000 + n_points)
    
    # col_1: 缓慢增长，但在第5个点有个巨大的正向噪声
    col_1 = np.linspace(10, 12, n_points)
    col_1[5] = 25.0  # 离群噪声点
    
    # col_2: 波动剧烈，但在第8个点有个负向噪声
    col_2 = np.linspace(20, 30, n_points)
    col_2[8] = 5.0   # 离群噪声点
    
    # col_3: 平稳数据
    col_3 = np.full(n_points, 50.0)
    col_3[3] = 65.0  # 离群噪声点
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'col_1': col_1,
        'col_2': col_2,
        'col_3': col_3
    })
    return df

# 2. 定义约束字典 (使用你指定的 (min, max) 格式)
# 这里的第二个值 (max) 将被提取作为 SMAX
sample_constraints = {
    'speed_constraints': {
        'col_1': (1.0, 2.0),   # SMAX = 2.0
        'col_2': (2.0, 5.0),   # SMAX = 5.0
        'col_3': (1.0, 3.0)    # SMAX = 3.0
    }
}

# 3. 执行清洗
if __name__ == "__main__":
    # 获取模拟数据
    raw_data = generate_sample_data()
    
    print("--- 原始脏数据 ---")
    print(raw_data)
    
    # 调用封装好的清洗函数
    # 注意：确保之前定义的 MTCSC_N 类和 _clean_with_mtcsc_N 函数在当前作用域内
    cleaned_df = _clean_with_mtcsc_N(raw_data, sample_constraints)
    
    print("\n--- 清洗后的数据 ---")
    print(cleaned_df)

    # 4. 验证清洗效果
    print("\n--- 异常修复验证 ---")
    for col in ['col_1', 'col_2', 'col_3']:
        diff = np.abs(raw_data[col] - cleaned_df[col]).sum()
        if diff > 0:
            print(f"列 {col}: 已成功修复异常波动。")