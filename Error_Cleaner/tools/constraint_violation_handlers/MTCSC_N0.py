import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import copy
import math


class TimePointN:
    def __init__(self, timestamp: float, values: Dict[str, float]):
        self.timestamp: float = timestamp
        self.values: Dict[str, float] = values.copy()
        self.modify: Dict[str, float] = values.copy()

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_modify(self) -> Dict[str, float]:
        return self.modify.copy()

    def set_modify(self, new_values: Dict[str, float]) -> None:
        self.modify.update(new_values)

    def __repr__(self):
        return f"TimePointN(ts={self.timestamp}, modify={self.modify})"


class TimeSeriesN:
    def __init__(self):
        self.timeseries: List[TimePointN] = []

    def add_point(self, point: TimePointN) -> None:
        self.timeseries.append(point)

    def get_timeseries(self) -> List[TimePointN]:
        return self.timeseries

    def get_length(self) -> int:
        return len(self.timeseries)

    def __len__(self):
        return len(self.timeseries)

    def __repr__(self):
        return f"TimeSeriesN({len(self.timeseries)} points)"


class MTCSC_N:
    """
    多变量时间序列速度约束修复器
    支持直接输入 pd.DataFrame
    """

    def __init__(
        self,
        timeseries,
        speed_constraints: Dict[str, tuple],
        t_window: float,
        timestamp_col: str = 'timestamp'
    ):
        """
        :param timeseries: pd.DataFrame 或 TimeSeriesN 对象
        :param speed_constraints: 变量速度约束字典，如 {'lon': (-0.8, 0.8), 'speed': (0, 20)}
        :param t_window: 滑动窗口大小（秒）
        :param timestamp_col: 时间戳列名（仅当输入为 DataFrame 时使用）
        """
        self.speed_constraints = speed_constraints
        self.T = t_window
        self.timestamp_col = timestamp_col
        self.variables = list(speed_constraints.keys())

        # 新增：支持 DataFrame 输入
        if isinstance(timeseries, pd.DataFrame):
            self.timeseries = self._df_to_timeseries(timeseries)
        elif isinstance(timeseries, TimeSeriesN):
            self.timeseries = timeseries
        else:
            raise TypeError("timeseries must be pd.DataFrame or TimeSeriesN")

        self.kp: Optional[TimePointN] = None

    def _df_to_timeseries(self, df: pd.DataFrame) -> TimeSeriesN:
        """将 DataFrame 转换为 TimeSeriesN"""
        ts = TimeSeriesN()
        # 确保按时间排序
        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        for _, row in df.iterrows():
            timestamp = float(row[self.timestamp_col])
            values = {
                col: float(row[col])
                for col in df.columns
                if col != self.timestamp_col and col in self.variables
            }
            # 检查变量是否缺失
            missing = set(self.variables) - set(values.keys())
            if missing:
                raise ValueError(f"Missing columns in DataFrame: {missing}")
            tp = TimePointN(timestamp=timestamp, values=values)
            ts.add_point(tp)
        return ts

    def main_screen(self) -> pd.DataFrame:
        """
        执行修复，并返回修复后的 DataFrame（更方便使用）
        """
        total_list: List[TimePointN] = self.timeseries.get_timeseries()
        size: int = len(total_list)

        if size == 0:
            return pd.DataFrame()

        pre_end: float = -1
        w_start_time: float = 0
        w_end_time: float = 0
        w_goal_time: float = 0
        cur_time: float = 0

        pre_point: Optional[TimePointN] = None
        tp: Optional[TimePointN] = None

        temp_series: TimeSeriesN = TimeSeriesN()
        temp_list: List[TimePointN]

        read_index: int = 1

        # 初始化第一个点
        tp = total_list[0]
        temp_series.add_point(tp)
        w_start_time = tp.get_timestamp()
        w_end_time = w_start_time
        w_goal_time = w_start_time + self.T

        while read_index < size:
            tp = total_list[read_index]
            cur_time = tp.get_timestamp()

            if cur_time > w_goal_time:
                while True:
                    temp_list = temp_series.get_timeseries()
                    if len(temp_list) == 0:
                        temp_series.add_point(tp)
                        w_goal_time = cur_time + self.T
                        w_end_time = cur_time
                        break

                    self.kp = temp_list[0]
                    w_start_time = self.kp.get_timestamp()
                    w_goal_time = w_start_time + self.T

                    if cur_time <= w_goal_time:
                        temp_series.add_point(tp)
                        w_end_time = cur_time
                        break

                    cur_end = w_end_time
                    if pre_end == -1:
                        pre_point = self.kp

                    self.local(temp_series, pre_point)
                    pre_point = self.kp
                    pre_end = cur_end

                    temp_series.get_timeseries().pop(0)
            else:
                if cur_time > w_end_time:
                    temp_series.add_point(tp)
                    w_end_time = cur_time

            read_index += 1

        # 处理最后一个窗口
        while temp_series.get_length() > 0:
            temp_list = temp_series.get_timeseries()
            self.kp = temp_list[0]
            if pre_point is None and temp_list:
                pre_point = self.kp
            if pre_point:
                self.local(temp_series, pre_point)
            pre_point = self.kp
            temp_list.pop(0)

        # 修复完成后，返回 DataFrame
        return self._timeseries_to_df()

    def _timeseries_to_df(self) -> pd.DataFrame:
        """将修复后的 TimeSeriesN 转回 DataFrame"""
        data = []
        for pt in self.timeseries.get_timeseries():
            row = {'timestamp': pt.get_timestamp()}
            row.update(pt.get_modify())
            data.append(row)
        return pd.DataFrame(data).sort_values(by='timestamp').reset_index(drop=True)

    def is_speed_valid(self, p1: TimePointN, p2: TimePointN) -> bool:
        dt = p2.get_timestamp() - p1.get_timestamp()
        if dt <= 0:
            return False
        for var in self.variables:
            v1 = p1.modify.get(var, 0)
            v2 = p2.modify.get(var, 0)
            rate = (v2 - v1) / dt
            min_rate, max_rate = self.speed_constraints[var]
            if rate < min_rate or rate > max_rate:
                return False
        return True

    def judge_modify(self, pre_point: TimePointN, max_point: TimePointN, kp: TimePointN) -> bool:
        return not (self.is_speed_valid(pre_point, kp) and self.is_speed_valid(kp, max_point))

    def local(self, time_series: TimeSeriesN, pre_point: TimePointN) -> None:
        temp_list: List[TimePointN] = time_series.get_timeseries()
        length: int = len(temp_list)
        if length == 0 or self.kp is None:
            return

        kp_time = self.kp.get_timestamp()
        top = [-1] * length
        chain_len = [0] * length
        start_idx = -1

        for i in range(length):
            tp = temp_list[i]
            if self.is_speed_valid(pre_point, tp):
                chain_len[i] = 1
                top[i] = -1
                start_idx = i
                break
        if start_idx == -1:
            return

        for i in range(start_idx + 1, length):
            tp_i = temp_list[i]
            for j in range(i - 1, start_idx - 1, -1):
                tp_j = temp_list[j]
                if chain_len[j] > 0 and self.is_speed_valid(tp_j, tp_i):
                    if chain_len[j] + 1 > chain_len[i]:
                        chain_len[i] = chain_len[j] + 1
                        top[i] = j
                    break

        max_idx = start_idx
        for i in range(start_idx, length):
            if chain_len[i] > chain_len[max_idx]:
                max_idx = i
        max_point = temp_list[max_idx]

        if self.judge_modify(pre_point, max_point, self.kp):
            pre_time = pre_point.get_timestamp()
            max_time = max_point.get_timestamp()
            dt_total = max_time - pre_time
            dt_kp_pre = kp_time - pre_time
            ratio = dt_kp_pre / dt_total if dt_total > 0 else 0.0

            new_values = {}
            for var in self.variables:
                v_pre = pre_point.modify.get(var, 0)
                v_max = max_point.modify.get(var, 0)
                repaired_val = v_pre + ratio * (v_max - v_pre)
                new_values[var] = repaired_val

            self.kp.set_modify(new_values)


if __name__ == '__main__':

    # 创建测试数据
    data = pd.DataFrame({
        'timestamp': [0, 1, 3, 5, 7, 10],
        'lon': [0.0, 0.5, 0.6, 1.0, 1.2, 2.0],
        'lat': [0.0, 2.0, 0.4, 0.8, 0.9, 1.0],
        'speed': [0.0, 50.0, 5.0, 10.0, 8.0, 12.0]
    })

    # 定义约束
    speed_constraints = {
        'lon': (-0.8, 0.8),
        'lat': (-0.6, 0.6),
        'speed': (0.0, 20.0)
    }

    # 直接用 DataFrame 初始化，自动处理！
    repairer = MTCSC_N(
        timeseries=data,
        speed_constraints=speed_constraints,
        t_window=4.0,
        timestamp_col='timestamp'
    )

    # 一行代码修复，返回 DataFrame
    repaired_df = repairer.main_screen()

    print(repaired_df.round(2))