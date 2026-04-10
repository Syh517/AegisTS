import numpy as np
import pandas as pd
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

@dataclass
class RepairNode:
    """
    表示一个待修复的点
    index: 在序列中的位置（int）
    u: 当前的速度变化量（加速度）
    """
    index: int
    u: float

    def __lt__(self, other):
        # heapq 是最小堆，我们要最大堆（按 |u - center| 降序），所以反过来比较
        return abs(self.u - Lsgreedy.center) > abs(other.u - Lsgreedy.center)

class Lsgreedy:
    # 类变量，用于 RepairNode 比较（避免传参）
    center = 0.0
    sigma = 1.0
    eps = 1e-12

    def __init__(self, df: pd.DataFrame, center: float = 0.0):
        """
        :param df: 时间序列 DataFrame，index 为时间戳（datetime 或 numeric），列是多变量
        :param center: 速度变化的期望中心，默认 0
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if df.isnull().values.any():
            warnings.warn("Input contains NaN values. They will be forward-filled.")
            df = df.fillna(method='ffill').fillna(method='bfill')

        self.df = df.copy()
        self.center = center
        self.cols = df.columns.tolist()
        self.n_cols = len(self.cols)
        self.time_index = df.index
        self.time_numeric = self._to_numeric_time(self.time_index)

        # 设置类变量（供 RepairNode 使用）
        Lsgreedy.center = center

    def _to_numeric_time(self, index) -> np.ndarray:
        """将时间索引转为数值（单位：秒）"""
        if pd.api.types.is_datetime64_any_dtype(index):
            return index.astype('datetime64[ns]').astype('int64') // 1_00_000_000  # 转为 0.1s 单位
        else:
            return index.values.astype(float)

    def _speed(self, values: np.ndarray, time: np.ndarray) -> np.ndarray:
        """计算速度 v[i] = (x[i+1] - x[i]) / (t[i+1] - t[i])"""
        dt = np.diff(time)
        dx = np.diff(values)
        # 防止除零
        dt = np.where(np.abs(dt) < self.eps, self.eps, dt)
        return dx / dt

    def _variation(self, speed: np.ndarray) -> np.ndarray:
        """计算速度变化（加速度）"""
        return np.diff(speed)

    def _median(self, arr: np.ndarray) -> float:
        return np.median(arr)

    def _mad(self, arr: np.ndarray) -> float:
        """MAD = 1.4826 * median(|x - median(x)|)"""
        med = self._median(arr)
        return 1.4826 * self._median(np.abs(arr - med))

    def _set_parameters(self, values: np.ndarray):
        """基于速度变化估计 sigma"""
        speed = self._speed(values, self.time_numeric)
        speed_change = self._variation(speed)
        self.sigma = self._mad(speed_change) if len(speed_change) > 0 else 1.0
        Lsgreedy.sigma = self.sigma

    def _repair_single_series(self, values: np.ndarray) -> np.ndarray:
        """
        修复单个时间序列
        :param values: 原始值数组
        :return: 修复后的数组
        """
        n = len(values)
        if n < 3:
            return values.copy()

        # 初始化修复数组
        repaired = values.copy()
        sigma = self.sigma or 1.0

        # 构建 RepairNode 初始堆（只处理中间点）
        heap: List[RepairNode] = []
        table = [None] * n  # table[i] 存储当前第 i 个点的 RepairNode

        for i in range(1, n - 1):
            node = self._create_node(i, repaired, self.time_numeric)
            table[i] = node
            if abs(node.u - self.center) > 3 * sigma:
                heapq.heappush(heap, node)

        # 贪心修复
        while heap:
            top = heap[0]  # 查看堆顶
            if abs(top.u - self.center) < max(self.eps, 3 * sigma):
                break

            # 弹出并修改
            heapq.heappop(heap)
            self._modify_node(top.index, repaired, self.time_numeric)

            # 更新邻居：index-1, index, index+1
            for j in range(max(1, top.index - 1), min(n - 1, top.index + 2)):
                # 移除旧节点（如果在堆中，我们不清除，而是靠 lazy 更新）
                # 这里采用 lazy heap：允许重复节点，只处理最新的
                new_node = self._create_node(j, repaired, self.time_numeric)
                table[j] = new_node
                if abs(new_node.u - self.center) > 3 * sigma:
                    heapq.heappush(heap, new_node)

        return repaired

    def _create_node(self, i: int, repaired: np.ndarray, time: np.ndarray) -> RepairNode:
        """创建 RepairNode"""
        v1 = (repaired[i + 1] - repaired[i]) / (time[i + 1] - time[i])
        v2 = (repaired[i] - repaired[i - 1]) / (time[i] - time[i - 1])
        u = v1 - v2
        return RepairNode(index=i, u=u)

    def _modify_node(self, idx: int, repaired: np.ndarray, time: np.ndarray):
        """修改 repaired[idx] 的值"""
        u = self._create_node(idx, repaired, time).u
        sigma = self.sigma or 1.0

        if sigma < self.eps:
            temp = abs(u - self.center)
        else:
            temp = max(sigma, abs(u - self.center) / 3)

        dt_forward = time[idx + 1] - time[idx]
        dt_backward = time[idx] - time[idx - 1]
        dt_total = time[idx + 1] - time[idx - 1]

        # 加权因子
        weight = (dt_forward * dt_backward) / (dt_total + self.eps)

        delta = temp * weight

        if u > self.center:
            repaired[idx] += delta
        else:
            repaired[idx] -= delta

    def repair(self) -> pd.DataFrame:
        """
        执行修复，返回修复后的 DataFrame
        """
        repaired_data = np.zeros_like(self.df.values, dtype=float)

        for col_idx, col in enumerate(self.cols):
            values = self.df[col].values
            self._set_parameters(values)  # 估计 sigma
            repaired_col = self._repair_single_series(values)
            repaired_data[:, col_idx] = repaired_col

        # 构造结果 DataFrame
        result_df = pd.DataFrame(repaired_data, index=self.df.index, columns=self.df.columns)
        return result_df
    

if __name__ == '__main__':
    # 假设时间戳是连续整数
    time_index = np.arange(10)

    # 构造一条序列，有一个明显异常点
    data = {
        "sensor1": np.array([1, 1.2, 1.1, 10.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),  # 第4个点异常
        "sensor2": np.array([2, 2.1, 2.0, 2.2, 2.1, 20.0, 2.3, 2.4, 2.5, 2.6])   # 第6个点异常
    }

    df = pd.DataFrame(data, index=time_index)
    print("原始数据:")
    print(df)

    lsg = Lsgreedy(df, center=0.0)
    repaired_df = lsg.repair()

    print("\n修复后的数据:")
    print(repaired_df)