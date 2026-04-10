import pandas as pd
import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Union


class TimePoint2:
    def __init__(self, timestamp: int, orgval: Dict[str, float], modify: Optional[Dict[str, float]] = None):
        self.timestamp = timestamp
        self.orgval = dict(orgval)  # {var: value}
        self.modify = dict(modify) if modify is not None else dict(orgval)

    def getOrgval(self) -> Dict[str, float]:
        return self.orgval

    def getModify(self) -> Dict[str, float]:
        return self.modify

    def setModify(self, new_modify: Dict[str, float]):
        self.modify = dict(new_modify)


class TimeSeries2:
    def __init__(self):
        self.timeseries: List[TimePoint2] = []

    def addPoint(self, point: TimePoint2):
        self.timeseries.append(point)

    def getTimeseries(self) -> List[TimePoint2]:
        return self.timeseries

    def getLength(self) -> int:
        return len(self.timeseries)


class MTCSC_AS:
    def __init__(self, df: pd.DataFrame, speed_constraints: Dict[str, Tuple[float, float]],
                 t: int,
                 drate: float = 0.025,
                 threshold: float = 0.5,
                 swsize: int = 200,
                 beta: float = 0.95):
        """
        多变量时序数据修复，基于速度约束

        :param df: DataFrame with 'timestamp' and other variable columns
        :param speed_constraints: dict like {"var1": (min_speed, max_speed), ...}
        :param t: sliding window duration (time units)
        :param drate: drift rate for adaptive threshold (not used yet, future extension)
        :param threshold: threshold for change detection (e.g., KL or other)
        :param swsize: size of speed history window
        :param beta: decay factor for adaptive update
        """
        # 检查列
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")

        self.variables = [col for col in df.columns if col != 'timestamp']
        for var in speed_constraints.keys():
            if var not in self.variables:
                raise ValueError(f"Speed constraint defined for '{var}' but not found in DataFrame columns")

        # 转换为时间序列
        self.timeseries = TimeSeries2()
        for _, row in df.iterrows():
            orgval = {var: float(row[var]) for var in self.variables}
            tp = TimePoint2(timestamp=int(row['timestamp']), orgval=orgval)
            self.timeseries.addPoint(tp)

        self.speed_constraints = speed_constraints  # {var: (min_dv_dt, max_dv_dt)}
        self.T = t
        self.Drate = drate
        self.Threshold = threshold
        self.SWSize = swsize
        self.Beta = beta

        # 为每个变量维护最近的速度历史（用于检测分布变化）
        self.speed_history: Dict[str, List[float]] = {var: [] for var in self.variables}
        self.now_speed_history: Dict[str, List[float]] = {var: [] for var in self.variables}

        # 是否启用自适应（未来可扩展）
        self.adaptive = False  # 当前先固定约束，后续可加

    def addSpeed(self, prePoint: TimePoint2, kpPoint: TimePoint2):
        """计算每个变量的变化速度，并更新历史记录"""
        time_diff = kpPoint.timestamp - prePoint.timestamp
        if time_diff <= 0:
            return

        for var in self.variables:
            if var not in self.speed_constraints:
                continue  # 无约束则跳过监控

            v1 = prePoint.getModify()[var]
            v2 = kpPoint.getModify()[var]
            speed = (v2 - v1) / time_diff

            # 添加到当前历史（now）
            if len(self.now_speed_history[var]) < self.SWSize:
                self.now_speed_history[var].append(speed)
            else:
                # 已满，移除最老，加入最新
                self.speed_history[var] = self.now_speed_history[var].copy()
                self.now_speed_history[var] = self.now_speed_history[var][1:] + [speed]

                # 可在此处触发 KL 检测或其它漂移检测
                # self.detect_drift(var)

    def detect_drift(self, var: str):
        """可选：使用 KL 散度检测速度分布漂移（需离散化）"""
        # 简化：直方图比较（略）
        pass

    @staticmethod
    def is_within_speed(val1: float, val2: float, dt: float, min_speed: float, max_speed: float) -> bool:
        if dt <= 0:
            return True
        dv_dt = (val2 - val1) / dt
        return min_speed <= dv_dt <= max_speed

    def judgeModify(self, preVal: Dict[str, float], maxVal: Dict[str, float], kpVal: Dict[str, float],
                    preTime: int, maxTime: int, kpTime: int) -> bool:
        """
        判断 kp 是否需要修改：是否违反从 pre 或 max 出发的速度约束
        """
        dt_kp_pre = kpTime - preTime
        dt_max_kp = maxTime - kpTime
        dt_max_pre = maxTime - preTime

        if dt_kp_pre <= 0 or dt_max_kp <= 0:
            return False

        need_modify = False
        for var in self.variables:
            if var not in self.speed_constraints:
                continue

            min_spd, max_spd = self.speed_constraints[var]

            # 条件1：pre -> kp 是否超速
            cond1 = self.is_within_speed(preVal[var], kpVal[var], dt_kp_pre, min_spd, max_spd)

            # 条件2：kp -> max 是否超速
            cond2 = self.is_within_speed(kpVal[var], maxVal[var], dt_max_kp, min_spd, max_spd)

            if not (cond1 and cond2):
                need_modify = True
                break

        return need_modify

    def local(self, timeSeries: TimeSeries2, prePoint: TimePoint2):
        tempList = timeSeries.getTimeseries()
        length = len(tempList)
        if length == 0:
            return

        kp = tempList[0]  # 当前窗口第一个点（可能被修改）
        preTime = prePoint.timestamp
        preVal = prePoint.getModify()
        kpTime = kp.timestamp
        kpVal = kp.getModify()

        if length == 1:
            if self.judgeModify(preVal, preVal, kpVal, preTime, preTime, kpTime):
                # 修正 kp -> 使用 preVal
                kp.setModify(dict(preVal))
            return

        # 动态规划：找最长合法链
        top = [-1] * length        # top[i] 表示 i 的前驱索引
        chain_len = [0] * length   # chain_len[i] 表示以 i 结尾的链长度

        # 初始化：找第一个可以从 prePoint 到达的点
        start_idx = -1
        for i in range(length):
            tp = tempList[i]
            t_diff = tp.timestamp - preTime
            if t_diff <= 0:
                continue
            valid = True
            for var in self.variables:
                if var not in self.speed_constraints:
                    continue
                min_spd, max_spd = self.speed_constraints[var]
                if not self.is_within_speed(preVal[var], tp.getModify()[var], t_diff, min_spd, max_spd):
                    valid = False
                    break
            if valid:
                top[i] = -1  # 来自 prePoint
                chain_len[i] = 1
                start_idx = i
                break

        if start_idx == -1:
            # 没有任何点可达，无法形成链
            return

        # 扩展链
        for i in range(start_idx + 1, length):
            tp_i = tempList[i]
            t_i = tp_i.timestamp
            valid = False
            for j in range(i - 1, start_idx - 1, -1):
                tp_j = tempList[j]
                if top[j] == -1 or top[j] >= 0:  # j 是链中一点
                    t_j = tp_j.timestamp
                    dt = t_i - t_j
                    if dt <= 0:
                        continue
                    inner_valid = True
                    for var in self.variables:
                        if var not in self.speed_constraints:
                            continue
                        min_spd, max_spd = self.speed_constraints[var]
                        if not self.is_within_speed(tp_j.getModify()[var], tp_i.getModify()[var], dt, min_spd, max_spd):
                            inner_valid = False
                            break
                    if inner_valid:
                        chain_len[i] = chain_len[j] + 1
                        top[i] = j
                        valid = True
                        break
            if not valid:
                # 尝试从 prePoint 直达
                dt = t_i - preTime
                if dt > 0:
                    direct_valid = True
                    for var in self.variables:
                        if var not in self.speed_constraints:
                            continue
                        min_spd, max_spd = self.speed_constraints[var]
                        if not self.is_within_speed(preVal[var], tp_i.getModify()[var], dt, min_spd, max_spd):
                            direct_valid = False
                            break
                    if direct_valid:
                        chain_len[i] = 1
                        top[i] = -1

        # 找最长链的终点
        max_idx = start_idx
        for i in range(start_idx, length):
            if chain_len[i] > chain_len[max_idx]:
                max_idx = i

        maxPoint = tempList[max_idx]
        maxVal = maxPoint.getModify()
        maxTime = maxPoint.timestamp

        # 判断是否需要修改 kp
        if self.judgeModify(preVal, maxVal, kpVal, preTime, maxTime, kpTime):
            # 修正 kp 的值
            new_modify = {}
            for var in self.variables:
                if var not in self.speed_constraints:
                    new_modify[var] = kpVal[var]
                    continue

                min_spd, max_spd = self.speed_constraints[var]
                dt_pre_kp = kpTime - preTime
                dt_pre_max = maxTime - preTime
                dt_kp_max = maxTime - kpTime

                if dt_pre_kp <= 0:
                    new_modify[var] = kpVal[var]
                    continue

                # 投影策略：限制 kp 在 pre 和 max 的“速度锥”内
                lower_bound = preVal[var] + min_spd * dt_pre_kp
                upper_bound = preVal[var] + max_spd * dt_pre_kp

                # 同时考虑从 max 回推
                rev_lower = maxVal[var] - max_spd * dt_kp_max
                rev_upper = maxVal[var] - min_spd * dt_kp_max

                # 交集
                final_lower = max(lower_bound, rev_lower)
                final_upper = min(upper_bound, rev_upper)

                if final_lower <= final_upper:
                    # 取中间值
                    repaired = (final_lower + final_upper) / 2.0
                else:
                    # 无交集，优先满足 pre -> kp
                    repaired = np.clip(kpVal[var], lower_bound, upper_bound)

                new_modify[var] = repaired

            kp.setModify(new_modify)

    def mainScreen(self) -> pd.DataFrame:
        totalList = self.timeseries.getTimeseries()
        size = len(totalList)
        if size == 0:
            return pd.DataFrame(columns=['timestamp'] + self.variables)

        preEnd = -1
        wStartTime = 0
        wEndTime = 0
        wGoalTime = 0
        curTime = 0
        prePoint = None

        tempSeries = TimeSeries2()
        readIndex = 1

        # 初始化第一个点
        tp0 = totalList[0]
        tempSeries.addPoint(tp0)
        wStartTime = tp0.timestamp
        wEndTime = wStartTime
        wGoalTime = wStartTime + self.T

        while readIndex < size:
            tp = totalList[readIndex]
            curTime = tp.timestamp

            if curTime > wGoalTime:
                # 处理当前窗口
                while tempSeries.getLength() > 0:
                    tempList = tempSeries.getTimeseries()
                    if len(tempList) == 0:
                        break
                    kp = tempList[0]
                    if prePoint is None:
                        prePoint = kp
                    self.local(tempSeries, prePoint)
                    prePoint = kp
                    tempList.pop(0)

                # 重置窗口
                tempSeries.addPoint(tp)
                wStartTime = tp.timestamp
                wEndTime = tp.timestamp
                wGoalTime = wStartTime + self.T
            else:
                if curTime > wEndTime:
                    if tempSeries.getLength() > 0:
                        last_temp = tempSeries.getTimeseries()[-1]
                        self.addSpeed(last_temp, tp)
                    tempSeries.addPoint(tp)
                    wEndTime = curTime
            readIndex += 1

        # 处理剩余点
        while tempSeries.getLength() > 0:
            tempList = tempSeries.getTimeseries()
            if len(tempList) == 0:
                break
            kp = tempList[0]
            if prePoint is None:
                prePoint = kp
            self.local(tempSeries, prePoint)
            prePoint = kp
            tempList.pop(0)

        # 输出结果
        data = []
        for tp in self.timeseries.getTimeseries():
            row = {'timestamp': tp.timestamp}
            row.update(tp.modify)
            data.append(row)
        return pd.DataFrame(data)
    


if __name__ == '__main__':
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
    # 应用修复
    repairer = MTCSC_AS(
        df=df,
        speed_constraints=speed_constraints,
        t=5,           # 时间窗口长度
        swsize=5       # 速度历史窗口
    )

    result_df = repairer.mainScreen()

    print("原始数据：")
    print(df)
    print("\n修复后数据：")
    print(result_df)