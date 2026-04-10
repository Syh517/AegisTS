import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import random

# 可选：仅在需要 OLS 时导入
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TimePoint:
    """表示一个多变量时间点"""
    def __init__(self, timestamp: float, orgval: Dict[str, float], truth: Dict[str, float] = None):
        self.timestamp = float(timestamp)
        self.orgval = orgval  # 原始观测值，dict: {"temp": 20.1, "pressure": 101.3}
        self.truth = truth or {}
        self.modify = orgval.copy()  # 可修复的值
        self.label = True  # True: 正常，False: 异常

    def __repr__(self):
        return f"TimePoint(t={self.timestamp}, val={self.modify}, label={self.label})"


class TimeSeries:
    """时序集合"""
    def __init__(self, points: List[TimePoint] = None):
        self.timeseries: List[TimePoint] = points or []
        self.num = 0
        self.num_list: List[int] = []

    def get_timeseries(self) -> List[TimePoint]:
        return self.timeseries

    def set_num(self, num: int):
        self.num = num

    def set_num_list(self, num_list: List[int]):
        self.num_list = num_list

    def __len__(self) -> int:
        return len(self.timeseries)

    def __iter__(self):
        return iter(self.timeseries)

    def __getitem__(self, idx: int) -> TimePoint:
        return self.timeseries[idx]


class HTDClean:
    """
    多变量 HTD 异常检测与清洗算法
    支持 per-variable 速度约束：
        speed_constraints = {
            "temp": (-2, 2),
            "pressure": (-3, 3)
        }
    """

    def __init__(
        self,
        speed_constraints: Dict[str, Tuple[float, float]],
        seed: int = 1,
        max_num: int = 5000,
        poly_degree: int = 3,
        use_ols: bool = True
    ):
        """
        :param speed_constraints: dict, 变量名 -> (min_speed, max_speed)
        :param seed: 随机种子（用于模拟噪声容忍）
        :param max_num: 预留参数（兼容原始 HTD）
        :param poly_degree: 多项式回归阶数（仅当 use_ols=True 时生效）
        :param use_ols: 是否使用 OLS 回归修复（否则用线性插值）
        """
        self.speed_constraints = speed_constraints
        self.seed = seed
        self.max_num = max_num
        self.poly_degree = poly_degree
        self.use_ols = use_ols and SKLEARN_AVAILABLE
        self.random_gen = random.Random(seed)
        self.timeseries: Optional[TimeSeries] = None
        self.size = 0
        self.variables = list(speed_constraints.keys())

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        # 构建 TimePoint 列表
        points = []
        for _, row in data.iterrows():
            orgval = {var: row[var] for var in self.variables}
            truth = {var: row.get(f"{var}_truth", None) for var in self.variables}
            truth = {k: v for k, v in truth.items() if v is not None}
            pt = TimePoint(
                timestamp=row['timestamp'],
                orgval=orgval,
                truth=truth
            )
            points.append(pt)

        self.timeseries = TimeSeries(points)
        self.size = len(self.timeseries)

        # 步骤1：异常检测
        outlier_indices = self._outlier_detection()

        # 步骤2：打标签 + 修复
        dirty_series = self._outlier_repair(outlier_indices)

        # 构建输出：只保留 timestamp 和 modify 后的变量值
        result = []
        for tp in dirty_series.get_timeseries():
            record = {'timestamp': tp.timestamp}
            for var in self.variables:
                record[var] = tp.modify[var]
            result.append(record)

        # 按时间排序并返回
        return pd.DataFrame(result).sort_values(by='timestamp').reset_index(drop=True)

    def _outlier_detection(self) -> List[int]:
        total_list = self.timeseries.get_timeseries()
        size = self.size

        outlier_index = list(range(size))
        anomaly = [i for i in range(size)]
        normal = [-1] * size

        for j in range(1, size):
            for i in range(j):
                if self._judge_speed(total_list[i], total_list[j]):
                    cost = anomaly[i] + (j - i - 1)
                    if cost < anomaly[j]:
                        anomaly[j] = cost
                        normal[j] = i

        best_cost = anomaly[0] + size - 1
        best_end = 0
        for j in range(1, size):
            cost = anomaly[j] + (size - j - 1)
            if cost < best_cost:
                best_cost = cost
                best_end = j

        current = best_end
        while current != -1:
            if current in outlier_index:
                idx_to_remove = outlier_index.index(current)
                outlier_index.pop(idx_to_remove)
            current = normal[current]

        return sorted(outlier_index)

    def _judge_speed(self, i_point: TimePoint, j_point: TimePoint) -> bool:
        dt = j_point.timestamp - i_point.timestamp
        if dt <= 0:
            return False

        for var in self.variables:
            min_speed, max_speed = self.speed_constraints[var]
            dv = j_point.modify[var] - i_point.modify[var]
            speed = dv / dt
            if not (min_speed <= speed <= max_speed):
                return False
        return True

    def _outlier_repair(self, outlier_indices: List[int]) -> TimeSeries:
        label = [True] * self.size
        num = 0
        num_list = []

        for i in range(self.size):
            if i in outlier_indices:
                if self.random_gen.random() < 0.1:
                    label[i] = True
                    num += 1
                    num_list.append(i)
                else:
                    label[i] = False
            else:
                label[i] = True

        tp_list = []
        original_points = self.timeseries.get_timeseries()
        for i, point in enumerate(original_points):
            new_point = TimePoint(
                timestamp=point.timestamp,
                orgval=point.orgval.copy(),
                truth=point.truth.copy() if point.truth else {}
            )
            new_point.modify = point.modify.copy()
            new_point.label = label[i]
            tp_list.append(new_point)

        dirty_series = TimeSeries(tp_list)
        dirty_series.set_num(num)
        dirty_series.set_num_list(num_list)

        # 选择修复策略
        if self.use_ols:
            self._repair_with_ols(dirty_series, self.poly_degree)
        else:
            self._impute_multi_var(dirty_series)

        return dirty_series

    def _repair_with_ols(self, series: TimeSeries, poly_degree: int = 3):
        points = series.get_timeseries()
        n = len(points)
        timestamps = np.array([pt.timestamp for pt in points])

        for var in self.variables:
            values = np.array([pt.modify[var] for pt in points])
            labels = np.array([pt.label for pt in points])

            if not np.any(labels):
                continue

            t_train = timestamps[labels].reshape(-1, 1)
            y_train = values[labels]

            if len(t_train) <= poly_degree:
                pred_val = np.mean(y_train)
                for i in range(n):
                    if not labels[i]:
                        points[i].modify[var] = float(pred_val)
                continue

            try:
                model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
                model.fit(t_train, y_train)
                for i in range(n):
                    if not labels[i]:
                        t_pred = np.array([[timestamps[i]]])
                        pred = model.predict(t_pred)[0]
                        points[i].modify[var] = float(pred)
            except Exception:
                self._fallback_linear_interpolation_for_var(points, var, labels, timestamps)

    def _impute_multi_var(self, series: TimeSeries):
        """纯线性插值修复（无 sklearn 依赖）"""
        points = series.get_timeseries()
        n = len(points)
        timestamps = [pt.timestamp for pt in points]

        for i in range(n):
            if not points[i].label:
                for var in self.variables:
                    prev_val, prev_t = None, None
                    next_val, next_t = None, None

                    for j in range(i - 1, -1, -1):
                        if points[j].label:
                            prev_val = points[j].modify[var]
                            prev_t = timestamps[j]
                            break
                    for j in range(i + 1, n):
                        if points[j].label:
                            next_val = points[j].modify[var]
                            next_t = timestamps[j]
                            break

                    if prev_t is not None and next_t is not None and next_t != prev_t:
                        ratio = (timestamps[i] - prev_t) / (next_t - prev_t)
                        repaired = prev_val + ratio * (next_val - prev_val)
                    elif prev_t is not None:
                        repaired = prev_val
                    elif next_t is not None:
                        repaired = next_val
                    else:
                        repaired = points[i].modify[var]

                    points[i].modify[var] = repaired

    def _fallback_linear_interpolation_for_var(self, points, var, labels, timestamps):
        """OLS 失败时的回退"""
        n = len(points)
        for i in range(n):
            if not labels[i]:
                prev_val, prev_t = None, None
                next_val, next_t = None, None

                for j in range(i - 1, -1, -1):
                    if labels[j]:
                        prev_val = points[j].modify[var]
                        prev_t = timestamps[j]
                        break
                for j in range(i + 1, n):
                    if labels[j]:
                        next_val = points[j].modify[var]
                        next_t = timestamps[j]
                        break

                if prev_t is not None and next_t is not None and next_t != prev_t:
                    ratio = (timestamps[i] - prev_t) / (next_t - prev_t)
                    repaired = prev_val + ratio * (next_val - prev_val)
                elif prev_t is not None:
                    repaired = prev_val
                else:
                    repaired = next_val if next_t is not None else points[i].modify[var]

                points[i].modify[var] = repaired


# ======================
# 测试示例
# ======================
if __name__ == "__main__":
    # 定义速度约束
    speed_constraints = {
        "temp": (-2.0, 2.0),       # 温度变化不超过 ±2 单位/时间
        "pressure": (-3.0, 3.0)    # 压力变化不超过 ±3 单位/时间
    }

    # 创建测试数据（含异常）
    data = pd.DataFrame({
        'timestamp': [0, 1, 2, 3, 4, 5, 6],
        'temp':      [20, 21, 25, 22, 23, 24, 25],
        'pressure':  [100, 102, 110, 103, 104, 105, 106]
    })

    print("=== 使用 OLS 修复 ===")
    cleaner_ols = HTDClean(
        speed_constraints=speed_constraints,
        seed=42,
        poly_degree=3,
        use_ols=True  # 自动检测 sklearn 是否可用
    )
    cleaned_ols = cleaner_ols.clean(data)
    print("清洗后（OLS）:")
    print(cleaned_ols.round(2))

    print("\n=== 使用线性插值修复 ===")
    cleaner_interp = HTDClean(
        speed_constraints=speed_constraints,
        seed=42,
        use_ols=False  # 强制使用插值
    )
    cleaned_interp = cleaner_interp.clean(data)
    print("清洗后（插值）:")
    print(cleaned_interp.round(2))