import pandas as pd
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Datasets.load_dataset import load_single_dataset


class DataManager:
    def __init__(
        self, data, label=None, observed_data=None, abnormal_rate=0, task_type=None, seed=42
    ):
        """
        初始化数据管理器。
        :param dataset: 数据集。
        :param abnormal_rate: 标记数据的比例。
        :param task_type: 下游任务类型，如 'classification', 'forecasting', 'clustering' 等。
        :param seed: 随机种子，用于确保错误注入的一致性
        """

        self.task_type = task_type
        self.seed = seed

        if len(data.shape) == 2:
            self.clean_data_raw = np.expand_dims(data, axis=0)
        else:
            self.clean_data_raw = data.copy()

        self.label = label.copy() if label is not None else None
        self.flag = False
       
        # 存储每个sample的DataManager对象
        self.samples = []
        
        if observed_data is None:
            self.scale_factors = {}  # 用于存储每个sample的每列的缩放因子和最小值
            self.observed_data_restored = np.zeros_like(self.clean_data_raw)
            self.error_mask_restored = np.zeros_like(self.clean_data_raw, dtype=bool)
            
            # 为每个sample创建独立的处理对象
            for sample_idx in range(self.clean_data_raw.shape[0]):
                sample_data = self.clean_data_raw[sample_idx]
                
                # 为每个sample创建独立的处理对象，并传递seed
                sample_manager = SingleSampleDataManager(sample_data, abnormal_rate, seed=self.seed)
                self.samples.append(sample_manager)
                

        else:
            # 如果提供了observed_data，则使用它（主要用于加载已有脏数据）
            self.observed_data_restored = observed_data.copy()
            self.error_mask_restored = observed_data != self.clean_data_raw
            self.flag = True
            # 注意：当提供observed_data时，samples不会被创建或使用。


    def inject_errors(self, error_ratio, error_types=None, covered_attrs=None):
        """
        向所有样本数据中注入错误。
        :param error_ratio: 每个样本中要注入错误的时间点比例。
        :param error_types: 要注入的错误类型列表。
        :param covered_attrs: 可以注入错误的属性（列）索引集合。
        """
        print("-----Inject Errors...-----")

        if error_ratio > 1.0 or error_ratio < 0.0:
            raise ValueError("error_ratio must be in range (0, 1]")
        elif error_ratio == 0.0:
            return
        if error_types is None:
            raise ValueError("error_types must be a list")

        # 对每个sample分别注入错误
        for sample_idx, sample_manager in enumerate(self.samples):
            sample_manager.inject_errors(error_ratio, error_types, covered_attrs)
        
        self.observed_data_restored, self.error_mask_restored = self.get_dirty_data_restored()

        print("-----Inject finished-----\n\n")


    def get_dirty_data_restored(self):
        """获取原始三维格式的脏数据"""
        if self.flag:
            return self.observed_data_restored, self.error_mask_restored
        else:
            dirty_data_raw = np.zeros_like(self.clean_data_raw)
            error_mask = np.zeros_like(self.clean_data_raw, dtype=bool)
            for i, sample_manager in enumerate(self.samples):
                dirty_data_raw[i] = sample_manager.restore_original_scale(sample_manager.observed_data)
                error_mask[i] = sample_manager.error_mask
            return dirty_data_raw, error_mask


class SingleSampleDataManager:
    """处理单个sample的数据管理器"""
    
    def __init__(self, data, abnormal_rate=0, seed=42):
        # 创建局部随机状态对象，避免影响全局随机状态
        self.np_random = np.random.RandomState(seed)
        self.random_state = random.Random(seed) if seed is not None else random.Random(seed)
        
        self.clean_data = data.copy()
        self.scale_factors = {}
        self._adjust_scale()
        self.observed_data = self.clean_data.copy()
        self.error_mask = np.zeros_like(self.clean_data, dtype=bool)
        self.is_label = np.zeros_like(self.clean_data, dtype=bool)
        self.randomly_label_data(abnormal_rate)
        self.col_min_max = {}
        self.compute_column_min_max()

    def _adjust_scale(self):
        """
        调整数据的量纲，使每列的均值约为10，且没有负数。
        """
        for col in range(self.clean_data.shape[1]):
            # Check if column contains string data
            try:
                # Try to convert first few non-NaN values to float to check if it's numeric
                sample_vals = self.clean_data[:, col][:5]  # Check first 5 values
                sample_vals = [x for x in sample_vals if pd.notna(x)]  # Remove NaN values
                if len(sample_vals) > 0:
                    # Try converting to float - if this fails, it's likely a string column
                    for val in sample_vals:
                        float(val)
            except (ValueError, TypeError):
                # This is a string column (like timestamp), skip scaling
                self.scale_factors[col] = (0, 1)  # No scaling for string columns
                continue
            
            # Process numeric columns
            try:
                min_val = np.min(self.clean_data[:, col].astype(float))
                self.clean_data[:, col] = self.clean_data[:, col].astype(float) - min_val
                mean_val = np.mean(self.clean_data[:, col])
                scale_factor = 10 / mean_val if mean_val != 0 else 1
                self.clean_data[:, col] *= scale_factor
                self.scale_factors[col] = (min_val, scale_factor)
            except (ValueError, TypeError):
                # If conversion fails, treat as string column
                self.scale_factors[col] = (0, 1)

    def compute_column_min_max(self):
        """
        计算 clean_data 中每一列的最大值和最小值，存入 self.col_min_max。
        """
        self.col_min_max = {}
        for col in range(self.clean_data.shape[1]):
            try:
                # Check if column is numeric
                float_vals = self.clean_data[:, col].astype(float)
                self.col_min_max[col] = (np.min(float_vals), np.max(float_vals))
            except (ValueError, TypeError):
                # For non-numeric columns, store None to indicate they shouldn't be processed
                self.col_min_max[col] = (None, None)

    def randomly_label_data(self, error_rate):
        # 确保error_rate在合理范围
        if not 0 <= error_rate <= 1:
            raise ValueError("Error rate must be between 0 and 1.")

        n_rows, n_cols = self.observed_data.shape
        total_elements = n_rows * n_cols
        n_errors = int(total_elements * error_rate)

        # 首先标记前3行为True
        self.is_label[:3, :] = True
        marked_elements = 3 * n_cols

        # 随机选择剩余要标记的数据单元
        remaining_elements = total_elements - marked_elements
        additional_errors = n_errors - marked_elements
        if additional_errors > 0:
            error_indices = self.np_random.choice(
                remaining_elements, additional_errors, replace=False
            )
            for idx in error_indices:
                # 调整索引以跳过已标记的前3行
                adjusted_idx = idx + marked_elements
                row, col = divmod(adjusted_idx, n_cols)
                self.is_label[row, col] = True

    def inject_errors(self, error_ratio, error_types, covered_attrs):
        """
        向数据中注入错误。
        :param error_ratio: 错误注入的比例。
        :param error_types: 要注入的错误类型列表。
        :param covered_attrs: 可以注入错误的属性集合。
        """
        n_rows, n_cols = self.clean_data.shape
        total_errors = int(n_rows * error_ratio)

        while total_errors > 0:
            error_length = self.random_state.randint(8, 16)
            if total_errors < error_length:  # 确保不超过剩余的错误数
                error_length = total_errors


            # 起始行的选择范围是 [20, n_rows - error_length]，这样注入的错误不会到达最后一行
            start_row = self.random_state.randint(20, n_rows - error_length)


            if self.error_mask[start_row : start_row + error_length].any():
                continue

            # 仅从 covered_attrs 中选择列进行错误注入
            selected_cols = self.random_state.sample(
                list(covered_attrs), k=self.random_state.randint(1, min(3, len(covered_attrs)))
            )
            
            # Filter error types based on column data types
            numeric_selected_cols = [col for col in selected_cols 
                                  if self.col_min_max[col][0] is not None and self.col_min_max[col][1] is not None]
            string_selected_cols = [col for col in selected_cols 
                                  if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None]
            
            # For string columns, only allow "missing", "missing_row" and "duplicate" error types
            filtered_error_types = []
            for error_type in error_types:
                if error_type in ["missing", "duplicate"]:
                    filtered_error_types.append(error_type)
                elif numeric_selected_cols:  # Only add other error types if there are numeric columns
                    filtered_error_types.append(error_type)
            
            if not filtered_error_types:
                continue
                
            error_type = self.random_state.choice(filtered_error_types)

            if error_type == "single":
                self._inject_single_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "drift":
                self._inject_drift_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "gaussian":
                self._inject_gaussian_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "volatility":
                self._inject_volatility_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "gradual":
                self._inject_gradual_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "sudden":
                self._inject_sudden_error(start_row, error_length, numeric_selected_cols)
            elif error_type == "missing":
                # Can be applied to both numeric and string columns
                self._inject_missing_error(start_row, error_length, selected_cols)
            elif error_type == "duplicate":
                # Can be applied to both numeric and string columns
                self._inject_duplicate_error(start_row, error_length)

            total_errors -= error_length

    def _inject_single_error(self, start_row, length, cols):
        """
        在指定的 start_row 行，对 cols 中每一列注入一个明显偏离的单点异常值。
        """
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            # 若该位置已被注入异常，则跳过
            if self.error_mask[start_row, col]:
                continue

            original_value = self.observed_data[start_row, col]
            col_min, col_max = self.col_min_max[col]

            # 设定异常值应与原值相差至少 30%
            min_required_diff = (col_max - col_min) * 0.3

            # 尝试找到一个合适的异常值
            for _ in range(100):
                injected_value = self.np_random.uniform(col_min, col_max)
                if abs(injected_value - original_value) >= min_required_diff:
                    break

            # 注入异常值，clip 保证在合理范围
            self.observed_data[start_row, col] = np.clip(injected_value, 0, 30)
            self.error_mask[start_row, col] = True

    def _inject_drift_error(self, start_row, length, cols):
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            # 随机选择漂移值的范围
            drift_value = 6

            temp_values = (
                self.observed_data[start_row : start_row + length, col].astype(float)
                + drift_value
            )
            # 确保数据在0到30的范围内
            self.observed_data[start_row : start_row + length, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_gaussian_error(self, start_row, length, cols):
        snr = 20  # 信噪比为20dB
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            signal_power = np.mean(self.clean_data[:, col].astype(float) ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = self.np_random.normal(0, np.sqrt(noise_power), length)
            temp_values = (
                self.observed_data[start_row : start_row + length, col].astype(float) + noise
            )
            # 确保数据在0到30的范围内
            self.observed_data[start_row : start_row + length, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_volatility_error(self, start_row, length, cols):
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            # 生成波动因子向量
            volatility_factors = self.np_random.uniform(0.7, 1.3, length)
            original_values = self.clean_data[
                start_row : start_row + length, col
            ].astype(float)
            temp_values = original_values * volatility_factors
            # 确保数据在0到30的范围内
            self.observed_data[start_row : start_row + length, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_gradual_error(self, start_row, length, cols):
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            # 随机选择错误的增增减方向和最终幅度
            direction = self.random_state.choice([-1, 1])
            magnitude = 6 * direction
            # 创建一个渐变的错误向量
            gradual_change = np.linspace(0, magnitude, length)
            # 在错误的最后一行进行迅速恢复
            gradual_change[-1] = 0

            temp_values = (
                self.observed_data[start_row : start_row + length, col].astype(float)
                + gradual_change
            )
            self.observed_data[start_row : start_row + length, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_sudden_error(self, start_row, length, cols):
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            # 随机选择错误的增减方向和最终幅度
            direction = self.random_state.choice([-1, 1])
            magnitude = 6 * direction
            # 创建一个突变的错误向量
            sudden_change = np.full(length, magnitude)
            # 计算恢复阶段的长度
            recovery_length = length - (length // 2)
            # 在错误段进行逐渐恢复
            sudden_change[-recovery_length:] = np.linspace(
                magnitude, 0, recovery_length
            )

            temp_values = (
                self.observed_data[start_row : start_row + length, col].astype(float)
                + sudden_change
            )
            self.observed_data[start_row : start_row + length, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_missing_error(self, start_row, length, cols):
        """
        在指定的行范围和列中注入缺失值（NaN）。
        :param start_row: 起始行索引。
        :param length: 注入缺失值的长度（行数）。
        :param cols: 要注入的列名列表。
        """
        for col in cols:
            # Skip injection for non-numeric columns
            if self.col_min_max[col][0] is None or self.col_min_max[col][1] is None:
                continue
                
            self.observed_data[start_row : start_row + length, col] = np.nan
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_missing_row_error(self, start_row, length):
        """
        在指定的行范围注入整行缺失值（所有列都设为NaN）。
        :param start_row: 起始行索引。
        :param length: 注入缺失值的长度（行数）。
        """
        n_cols = self.observed_data.shape[1]
        # 选择所有列
        all_cols = list(range(n_cols))
        
        for col in all_cols:
            # 对所有列都设置为NaN
            self.observed_data[start_row : start_row + length, col] = np.nan
            self.error_mask[start_row : start_row + length, col] = True

    def _inject_duplicate_error(self, start_row, length):
        """
        在指定起始行和长度范围内注入重复值异常。

        参数：
            start_row (int): 起始行索引。
            length (int): 注入重复值的长度。
            cols (list[str], optional): 指定列名。如果为 None，则使用所有列。

        效果：
            - 将指定列在 [start_row, start_row+length] 范围内替换为 start_row 对应行的 clean_data。
            - 更新 error_mask。
        """
        end_row = min(start_row + length, len(self.clean_data))  # 防止越界

        # 使用 numpy 数组索引而不是 pandas 的 .loc
        repeat_row = self.clean_data[start_row].copy()  # 复制起始行的数据
        
        # 从 start_row+1 行开始，将数据替换为重复值
        for i in range(start_row + 1, end_row):
            self.observed_data[i] = repeat_row
            self.error_mask[i] = True

    def restore_original_scale(self, data):
        """
        将数据复原到原始的量纲。
        :param data: 要复原的数据。
        :return: 复原后的数据。
        """
        restored_data = data.copy()
        for col in range(data.shape[1]):
            if col in self.scale_factors:
                min_val, scale_factor = self.scale_factors[col]
                # Skip restoration for string columns (where scale_factor would be 1 and min_val would be 0)
                if min_val != 0 or scale_factor != 1:
                    restored_data[:, col] = (restored_data[:, col].astype(float) / scale_factor) + min_val
        return restored_data


if __name__ == "__main__":
    data,label =load_single_dataset('classification', 'EthanolConcentration')
    print(data.shape, label.shape)
    dm = DataManager(data, label, abnormal_rate=0.1, task_type='classification', seed=42)
    print(dm.clean_data_raw.shape)
    dm.inject_errors(
        0.1,
        [
            "single",
            "drift",
            "gaussian",
            "volatility",
            "gradual",
            "sudden",
            "missing",
            # "missing_row",
            "duplicate",
        ],
        covered_attrs = range(data.shape[-1]) 
    )


    print(dm.observed_data_restored.shape)
    print(dm.error_mask_restored.shape)

    if np.array_equal(dm.clean_data_raw, dm.observed_data_restored):
        print("数据一致")
    else:
        print("数据不一致")


