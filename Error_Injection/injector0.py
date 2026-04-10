import pandas as pd
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



class DataManager:
    def __init__(
        self, dataset, clean_dataset_path, observed_dataset_path=None, label_rate=0.1,
    ):
        """
        初始化数据管理器。
        :param dataset: 数据库的名称。
        :param dataset_path: 数据集的文件路径。
        """
        self.dataset = dataset
        self.clean_data = pd.read_csv(clean_dataset_path)

        if observed_dataset_path == None:
            self.scale_factors = {}  # 用于存储每列的缩放因子和最小值
            self._adjust_scale()
            self.observed_data = self.clean_data.copy()
            self.error_mask = pd.DataFrame(
                False, index=self.clean_data.index, columns=self.clean_data.columns
            )
            self.is_label = pd.DataFrame(
                False,
                index=self.observed_data.index,
                columns=self.observed_data.columns,
            )  # 初始化is_label
            self.randomly_label_data(label_rate)  # 随机标记指定比例的数据
            self.col_min_max = {}
            self.compute_column_min_max()
        else:
            self.observed_data = pd.read_csv(observed_dataset_path)
            self.error_mask = self.observed_data != self.clean_data

    def estimate_kalman_parameters(self):
        # 针对每列估计参数
        initial_state = {}
        observation_covariance = {}
        transition_covariance = {}
        transition_matrices = {}

        for col in self.clean_data.columns:
            initial_state[col] = self.clean_data[col].iloc[0]
            var_col = np.var(self.clean_data[col])
            observation_covariance[col] = var_col if var_col > 0 else 1
            transition_covariance[col] = var_col / 2 if var_col > 0 else 0.5
            transition_matrices[col] = 1  # 单变量情况下，转换矩阵是标量 1

        return (
            initial_state,
            observation_covariance,
            transition_covariance,
            transition_matrices,
        )

    def randomly_label_data(self, error_rate):
        # 确保error_rate在合理范围
        if not 0 <= error_rate <= 1:
            raise ValueError("Error rate must be between 0 and 1.")

        n_rows, n_cols = self.observed_data.shape
        total_elements = n_rows * n_cols
        n_errors = int(total_elements * error_rate)

        # 首先标记前3行为True
        self.is_label.iloc[:3, :] = True
        marked_elements = 3 * n_cols

        # 随机选择剩余要标记的数据单元
        remaining_elements = total_elements - marked_elements
        additional_errors = n_errors - marked_elements
        if additional_errors > 0:
            error_indices = np.random.choice(
                remaining_elements, additional_errors, replace=False
            )
            for idx in error_indices:
                # 调整索引以跳过已标记的前3行
                adjusted_idx = idx + marked_elements
                row, col = divmod(adjusted_idx, n_cols)
                self.is_label.iloc[row, col] = True

    def _adjust_scale(self):
        """
        调整数据的量纲，使每列的均值约为10，且没有负数。
        """
        for col in self.clean_data.columns:
            min_val = self.clean_data[col].min()
            self.clean_data[col] -= min_val
            mean_val = self.clean_data[col].mean()
            scale_factor = 10 / mean_val
            self.clean_data[col] *= scale_factor
            self.scale_factors[col] = (min_val, scale_factor)

    def compute_column_min_max(self):
        """
        计算 clean_data 中每一列的最大值和最小值，存入 self.col_min_max。
        """
        self.col_min_max = {
            col: (self.clean_data[col].min(), self.clean_data[col].max())
            for col in self.clean_data.columns
        }

    def restore_original_scale(self, data):
        """
        将数据复原到原始的量纲。
        :param data: 要复原的数据。
        :return: 复原后的数据。
        """
        restored_data = data.copy()
        for col in restored_data.columns:
            min_val, scale_factor = self.scale_factors[col]
            restored_data[col] = (restored_data[col] / scale_factor) + min_val
        return restored_data

    def inject_errors(self, error_ratio, error_types, covered_attrs):
        """
        向数据中注入错误。
        :param error_ratio: 错误注入的比例。
        :param error_types: 要注入的错误类型列表。
        :param covered_attrs: 可以注入错误的属性集合。
        """
        print("-----Inject Errors...-----")

        n_rows, _ = self.clean_data.shape
        total_errors = int(n_rows * error_ratio)

        while total_errors > 0:
            error_length = random.randint(8, 16)
            if total_errors < error_length:  # 确保不超过剩余的错误数
                error_length = total_errors

            start_row = random.randint(
                100, n_rows - error_length
            )  # 不在前100行添加噪声
            if self.error_mask.iloc[start_row : start_row + error_length].any().any():
                continue

            # 仅从 covered_attrs 中选择列进行错误注入
            selected_cols = random.sample(
                list(covered_attrs), k=random.randint(1, min(3, len(covered_attrs)))
            )
            # selected_cols = random.sample(list(covered_attrs), k=1)
            error_type = random.choice(error_types)

            total_errors -= error_length

            if error_type == "single":
                self._inject_single_error(start_row, error_length, selected_cols)
            elif error_type == "drift":
                self._inject_drift_error(start_row, error_length, selected_cols)
            elif error_type == "gaussian":
                self._inject_gaussian_error(start_row, error_length, selected_cols)
            elif error_type == "volatility":
                self._inject_volatility_error(start_row, error_length, selected_cols)
            elif error_type == "gradual":
                self._inject_gradual_error(start_row, error_length, selected_cols)
            elif error_type == "sudden":
                self._inject_sudden_error(start_row, error_length, selected_cols)
            elif error_type == "missing":
                self._inject_missing_error(start_row, error_length, selected_cols)
            elif error_type == "duplicate":
                self._inject_duplicate_error(start_row, error_length)

            total_errors -= error_length

        # 在错误注入之后，更新复原后的数据
        self.restored_clean_data = self.restore_original_scale(self.clean_data)
        self.restored_observed_data = self.restore_original_scale(self.observed_data)

        # 将注入错误后的数据存入self.detect_data

        print("-----Inject finished-----\n\n")

    def _inject_single_error(self, start_row, length, cols):
        """
        在指定的 start_row 行，对 cols 中每一列注入一个明显偏离的单点异常值。
        """
        for col in cols:
            # 若该位置已被注入异常，则跳过
            if self.error_mask.loc[start_row, col]:
                continue

            original_value = self.observed_data.loc[start_row, col]
            col_min, col_max = self.col_min_max[col]

            # 设定异常值应与原值相差至少 30%
            min_required_diff = (col_max - col_min) * 0.3

            # 尝试找到一个合适的异常值
            for _ in range(100):
                injected_value = np.random.uniform(col_min, col_max)
                if abs(injected_value - original_value) >= min_required_diff:
                    break

            # 注入异常值，clip 保证在合理范围
            self.observed_data.loc[start_row, col] = np.clip(injected_value, 0, 30)
            self.error_mask.loc[start_row, col] = True

    def _inject_drift_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择漂移值的范围
            drift_value = 6

            temp_values = (
                self.observed_data.loc[start_row : start_row + length - 1, col]
                + drift_value
            )
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_gaussian_error(self, start_row, length, cols):
        snr = 20  # 信噪比为20dB
        for col in cols:
            signal_power = np.mean(self.clean_data[col] ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), length)
            temp_values = (
                self.observed_data.loc[start_row : start_row + length - 1, col] + noise
            )
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_volatility_error(self, start_row, length, cols):
        for col in cols:
            # 生成波动因子向量
            volatility_factors = np.random.uniform(0.7, 1.3, length)
            original_values = self.clean_data.loc[
                start_row : start_row + length - 1, col
            ]
            temp_values = original_values * volatility_factors
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_gradual_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择错误的增减方向和最终幅度
            direction = random.choice([-1, 1])
            magnitude = 6 * direction
            # 创建一个渐变的错误向量
            gradual_change = np.linspace(0, magnitude, length)
            # 在错误的最后一行进行迅速恢复
            gradual_change[-1] = 0

            temp_values = (
                self.observed_data.loc[start_row : start_row + length - 1, col]
                + gradual_change
            )
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_sudden_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择错误的增减方向和最终幅度
            direction = random.choice([-1, 1])
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
                self.observed_data.loc[start_row : start_row + length - 1, col]
                + sudden_change
            )
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.clip(
                temp_values, 0, 30
            )
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_missing_error(self, start_row, length, cols):
        """
        在指定的行范围和列中注入缺失值（NaN）。
        :param start_row: 起始行索引。
        :param length: 注入缺失值的长度（行数）。
        :param cols: 要注入的列名列表。
        """
        for col in cols:
            self.observed_data.loc[start_row : start_row + length - 1, col] = np.nan
            self.error_mask.loc[start_row : start_row + length - 1, col] = True

    def _inject_duplicate_error(self, start_row, length, cols=None):
        """
        在指定起始行和长度范围内注入重复值异常。

        参数：
            start_row (int): 起始行索引。
            length (int): 注入重复值的长度。
            cols (list[str], optional): 指定列名。如果为 None，则使用所有列。

        效果：
            - 将指定列在 [start_row, start_row+length) 范围内替换为 start_row 对应行的 clean_data。
            - 更新 error_mask。
        """

        end_row = min(start_row + length, len(self.clean_data))  # 防止越界

        repeat_row = self.clean_data.loc[start_row]
        # 从start_row后一行还是定义为重复值
        for i in range(start_row + 1, end_row):
            self.observed_data.loc[i, :] = repeat_row
            self.error_mask.loc[i, :] = True

        # # 列处理：默认所有列
        # if cols is None:
        #     cols = self.clean_data.columns.tolist()

        # for col in cols:
        #     if col not in self.clean_data.columns:
        #         continue  # 跳过非法列名

        #     repeat_value = self.clean_data.loc[start_row, col]
        #     self.observed_data.loc[start_row + 1:end_row - 1, col] = repeat_value
        #     self.error_mask.loc[start_row + 1:end_row - 1, col] = True

    def get_dirty_data(self):
        return self.restored_observed_data.copy(deep=True)


def plot_error_segments(dm, buffer=10):
    """
    遍历并绘制每列上每段错误数据及其前后缓冲区的数据。
    :param dm: DataManager实例。
    :param buffer: 在错误数据前后额外包含的行数。
    """
    for col in dm.error_mask.columns:
        # 寻找错误段
        error_locations = dm.error_mask[col]
        start = None
        for i in range(len(error_locations)):
            if error_locations[i] and start is None:
                start = i
            elif not error_locations[i] and start is not None:
                end = i
                plot_segment(dm, col, start, end, buffer)
                start = None
        if start is not None:  # 处理最后一个错误段
            plot_segment(dm, col, start, len(error_locations), buffer)


def plot_segment(dm, col, start, end, buffer):
    """
    绘制特定列上的错误段及其周围的数据。
    :param dm: DataManager实例。
    :param col: 列名。
    :param start: 错误段开始位置。
    :param end: 错误段结束位置。
    :param buffer: 在错误数据前后额外包含的行数。
    """
    start = max(0, start - buffer)
    end = min(len(dm.clean_data), end + buffer)

    plt.figure(figsize=(10, 4))
    plt.plot(
        dm.restored_clean_data[col][start:end],
        label="Restored Clean Data",
        color="blue",
    )
    plt.plot(
        dm.restored_observed_data[col][start:end],
        label="Restored Observed Data",
        color="orange",
    )
    plt.axvline(
        x=start + buffer - 1, color="green", linestyle="--", label="Error Start"
    )
    plt.axvline(x=end - buffer, color="red", linestyle="--", label="Error End")
    plt.title(f"Error Segment in Column {col}")
    plt.legend()
    plt.savefig("output.png")
    plt.show()


if __name__ == "__main__":
    dm = DataManager("idf", "/home/yyy/TSC/TSClean/AutoClean/Datasets/IDF_Power/IDF_Power_Clean.csv")
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
            "duplicate",
        ],
        covered_attrs=dm.clean_data.columns,
    )

    print(dm.restored_clean_data)
    print(dm.restored_observed_data)

    # # 绘制每个错误段数据
    # plot_error_segments(dm, buffer=10)

    # # 绘制复原后的数据对比
    # n_cols = len(dm.restored_clean_data.columns)
    # cols_per_group = 5

    # for i in range(0, n_cols, cols_per_group):
    #     end = min(i + cols_per_group, n_cols)
    #     plt.figure(figsize=(12, 10))

    #     for j, col in enumerate(dm.restored_clean_data.columns[i:end]):
    #         # 绘制复原后的clean_data
    #         plt.subplot(cols_per_group, 2, 2 * j + 1)
    #         plt.plot(
    #             dm.restored_clean_data[col], label="Restored Clean Data", color="blue"
    #         )
    #         plt.title(f"Restored Clean Data - {col}")
    #         plt.legend()

    #         # 绘制复原后的observed_data
    #         plt.subplot(cols_per_group, 2, 2 * j + 2)
    #         plt.plot(
    #             dm.restored_observed_data[col],
    #             label="Restored Observed Data",
    #             color="orange",
    #         )
    #         plt.title(f"Restored Observed Data - {col}")
    #         plt.legend()

    #     plt.tight_layout()
    #     plt.savefig("idf.png")
    #     plt.show()



    # 绘制调整量纲后的数据对比
    # dm = DataManager('idf', 'datasets/idf.csv')
    # dm.inject_errors(0.1, ['drift', 'gaussian', 'volatility'])
    #
    # # 将列分为每组5列，并为每组绘制图表（左侧clean_data，右侧observed_data）
    # n_cols = len(dm.clean_data.columns)
    # cols_per_group = 5
    #
    # for i in range(0, n_cols, cols_per_group):
    #     end = min(i + cols_per_group, n_cols)
    #     plt.figure(figsize=(12, 10))  # 调整窗口大小
    #
    #     for j, col in enumerate(dm.clean_data.columns[i:end]):
    #         # 绘制clean_data
    #         plt.subplot(cols_per_group, 2, 2*j+1)
    #         plt.plot(dm.clean_data[col], label='Clean Data', color='blue')
    #         plt.title(f'Clean Data - {col}')
    #         plt.legend()
    #
    #         # 绘制observed_data
    #         plt.subplot(cols_per_group, 2, 2*j+2)
    #         plt.plot(dm.observed_data[col], label='Observed Data', color='orange')
    #         plt.title(f'Observed Data - {col}')
    #         plt.legend()
    #
    #     plt.tight_layout()
    #     plt.show()