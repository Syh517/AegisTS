import pandas as pd
import numpy as np


class TimePoint:
    def __init__(self, timestamp, orgval):
        self.timestamp = timestamp
        self.orgval = float(orgval)
        self.modify = float(orgval)  # 初始化为原始值


class TimeSeries:
    def __init__(self, timeseries=None):
        self.timeseries = timeseries or []

    def get_timeseries(self):
        return self.timeseries


class SpeedAcc:
    def __init__(self, df, time_col, data_cols, T, speed_constraints, acc_constraints):
        """
        基于速度和加速度约束的时间序列修复类。

        :param df: pd.DataFrame, 原始数据
        :param time_col: str, 时间戳列名
        :param data_cols: list of str, 待处理的变量列
        :param T: float, 时间窗口大小（单位与时间戳一致）
        :param speed_constraints: dict, 如 {"temp": (-2, 2)}，表示 [min_speed, max_speed]
        :param acc_constraints: dict, 如 {"temp": (-0.5, 0.5)}，表示 [min_acc, max_acc]
        """
        # 排序并验证
        self.df = df.sort_values(by=time_col).reset_index(drop=True)
        self.time_col = time_col
        self.data_cols = data_cols
        self.T = T

        # 解析约束字典
        self.SMIN, self.SMAX = self._parse_bounds(speed_constraints, "speed_constraints")
        self.AMIN, self.AMAX = self._parse_bounds(acc_constraints, "acc_constraints")

    def _parse_bounds(self, constraints, name):
        """解析形如 {"x": (low, high)} 的约束字典，返回 (low_dict, high_dict)"""
        low_dict = {}
        high_dict = {}
        for col in self.data_cols:
            if col not in constraints:
                raise ValueError(f"Missing {name} for column '{col}'")
            bound = constraints[col]
            if not (isinstance(bound, (list, tuple)) and len(bound) == 2):
                raise ValueError(f"{name}['{col}'] must be tuple/list of (min, max)")
            low, high = bound
            if low > high:
                raise ValueError(f"In {name}['{col}']: min ({low}) > max ({high})")
            low_dict[col] = float(low)
            high_dict[col] = float(high)
        return low_dict, high_dict

    def _to_time_series(self, col_name):
        """将某一列转换为 TimeSeries 对象"""
        points = [
            TimePoint(row[self.time_col], row[col_name])
            for _, row in self.df.iterrows()
        ]
        return TimeSeries(points)

    def _apply_speedacc(self, time_series, col_name):
        """核心 SpeedAcc 算法，使用当前列的速度和加速度约束"""
        total_list = time_series.get_timeseries()
        total = len(total_list)

        # 获取当前列的约束
        SMAX = self.SMAX[col_name]
        SMIN = self.SMIN[col_name]
        AMAX = self.AMAX[col_name]
        AMIN = self.AMIN[col_name]

        # Step 1: 构建 kT 数组（滑动窗口右边界）
        kT = [0] * total
        kn = 0
        ki = kn + 1
        while kn < total and ki < total:
            if total_list[ki].timestamp - total_list[kn].timestamp > self.T:
                kT[kn] = ki
                kn += 1
                ki = kn + 1
            else:
                ki += 1
        for i in range(total):
            if kT[i] == 0:
                kT[i] = total - 1

        tpk = [tp for tp in total_list]  # 点列表

        # === 处理前两个点（仅速度约束）===
        for k in range(2):
            xK = []

            # 速度约束：来自前一点
            if k > 0:
                dt = tpk[k].timestamp - tpk[k-1].timestamp
                if dt != 0:
                    xkmin = tpk[k-1].modify + SMIN * dt
                    xkmax = tpk[k-1].modify + SMAX * dt
                else:
                    xkmin = xkmax = tpk[k-1].modify
            else:
                xkmin, xkmax = -float('inf'), float('inf')

            # 候选值：来自窗口内其他点的速度反推
            for i in range(k+1, kT[k]):
                dt_ki = tpk[k].timestamp - tpk[i].timestamp
                if abs(dt_ki) < 1e-9:
                    continue
                val_speed_min = tpk[i].orgval + SMIN * dt_ki
                val_speed_max = tpk[i].orgval + SMAX * dt_ki
                xK.append(val_speed_min)
                xK.append(val_speed_max)

            xK.append(tpk[k].orgval)
            if not xK:
                tpk[k].modify = tpk[k].orgval
            else:
                xK.sort()
                xKmid = xK[len(xK) // 2]
                if xKmid > xkmax:
                    tpk[k].modify = xkmax
                elif xKmid < xkmin:
                    tpk[k].modify = xkmin
                else:
                    tpk[k].modify = xKmid

        # === 处理 k >= 2 的点（加速度 + 速度联合约束）===
        for k in range(2, total):
            xK = []
            dt_k_k1 = tpk[k].timestamp - tpk[k-1].timestamp
            if abs(dt_k_k1) < 1e-9:
                tpk[k].modify = tpk[k].orgval
                continue

            # 上一步的速度（斜率）
            dt_k1_k2 = tpk[k-1].timestamp - tpk[k-2].timestamp
            if abs(dt_k1_k2) < 1e-9:
                prev_slope = 0.0
            else:
                prev_slope = (tpk[k-1].modify - tpk[k-2].modify) / dt_k1_k2

            # 加速度约束：xk = x_{k-1} + v_{k-1} * dt + 0.5 * a * dt^2
            # 近似：v_k = v_{k-1} + a * dt => v_k ∈ [prev_slope + AMIN*dt, prev_slope + AMAX*dt]
            # 则 xk ∈ [x_{k-1} + v_k * dt]
            v_min = prev_slope + AMIN * dt_k_k1
            v_max = prev_slope + AMAX * dt_k_k1
            xkmin_acc = tpk[k-1].modify + v_min * dt_k_k1
            xkmax_acc = tpk[k-1].modify + v_max * dt_k_k1

            # 速度约束（直接）
            xkmin_vel = tpk[k-1].modify + SMIN * dt_k_k1
            xkmax_vel = tpk[k-1].modify + SMAX * dt_k_k1

            # 联合约束：取交集
            xkmin = max(xkmin_acc, xkmin_vel)
            xkmax = min(xkmax_acc, xkmax_vel)

            # 候选值：从窗口内 (j,i) 点对反推（加速度 & 速度）
            for j in range(k+1, kT[k]):
                for i in range(j+1, kT[k]):
                    if total_list[i].timestamp - total_list[k].timestamp > self.T:
                        continue
                    dt_ij = tpk[i].timestamp - tpk[j].timestamp
                    if abs(dt_ij) < 1e-9:
                        continue
                    slope_ij = (tpk[i].orgval - tpk[j].orgval) / dt_ij
                    dt_jk = tpk[j].timestamp - tpk[k].timestamp

                    # 加速度反推：假设从 (j) 到 (k) 的加速度受限
                    v_jk_min = slope_ij + AMIN * dt_ij
                    v_jk_max = slope_ij + AMAX * dt_ij
                    val_acc_min = tpk[j].orgval + v_jk_min * dt_jk
                    val_acc_max = tpk[j].orgval + v_jk_max * dt_jk
                    xK.append(val_acc_min)
                    xK.append(val_acc_max)

                    # 速度反推：从 j 到 k
                    dt_jk_speed = tpk[k].timestamp - tpk[j].timestamp
                    val_speed_min = tpk[j].orgval + SMIN * dt_jk_speed
                    val_speed_max = tpk[j].orgval + SMAX * dt_jk_speed
                    xK.append(val_speed_min)
                    xK.append(val_speed_max)

            xK.append(tpk[k].orgval)
            if not xK:
                tpk[k].modify = tpk[k].orgval
            else:
                xK.sort()
                xKmid = xK[len(xK) // 2]
                if xKmid > xkmax:
                    tpk[k].modify = xkmax
                elif xKmid < xkmin:
                    tpk[k].modify = xkmin
                else:
                    tpk[k].modify = xKmid

        return [pt.modify for pt in tpk]

    def run(self):
        """执行修复并返回增强的 DataFrame"""
        result_df = self.df.copy()

        for col in self.data_cols:
            print(f"🔧 Processing '{col}' "
                  f"| Speed [{self.SMIN[col]:+.3f}, {self.SMAX[col]:+.3f}] "
                  f"| Acc [{self.AMIN[col]:+.3f}, {self.AMAX[col]:+.3f}]")
            ts = self._to_time_series(col)
            modified_values = self._apply_speedacc(ts, col)
            result_df[f"{col}_clean"] = modified_values

        return result_df
    
if __name__ == "__main__":
    # 构造测试数据
    df = pd.DataFrame({
        'time': np.arange(0, 20, 1),
        'temp': 20 + 0.5 * np.arange(20) + np.random.normal(0, 0.2, 20),
        'pressure': 100 + 0.1 * np.arange(20) + np.random.normal(0, 0.1, 20)
    })

    # 注入异常
    df.loc[10, 'temp'] = 50  # 突变
    df.loc[15, 'pressure'] = 200

    # 定义约束
    speed_constraints = {
        "temp": (-2, 2),       # 温度变化速度：±2 单位/时间
        "pressure": (-3, 3)
    }
    acc_constraints = {
        "temp": (-0.5, 0.5),   # 加速度限制
        "pressure": (-0.7, 0.7)
    }

    # 执行修复
    sa = SpeedAcc(
        df=df,
        time_col='time',
        data_cols=['temp', 'pressure'],
        T=5,  # 5 时间单位的窗口
        speed_constraints=speed_constraints,
        acc_constraints=acc_constraints
    )

    result = sa.run()

    # 查看结果
    print(result[['temp', 'temp_clean', 'pressure', 'pressure_clean']])