# -*- coding: utf-8 -*-
# @Author  : Genglong Li
# @Time    : 2023/6/15 15:53
# @Comment : Benchmark Data Cleaning Algorithms for Time-Series Data
import numpy as np
from scipy.optimize import linprog
from statsmodels.tsa.ar_model import AutoReg
from Error_Cleaner.tools.constraint_violation_handlers.waste.Clean4MTS_ import Clean4MTS
import pandas as pd


# def solve_quad(A, B, C):
#     delta = B ** 2 - 4 * A * C  # 计算delta
#     if delta <= 0:  # 判断delta如果小于0
#         x = B / (-2 * A)  # 则解为B / (-2*A)
#         return x, x
#     else:  # 除以上两种情况以外，即delta如果大于0
#         x1 = (B + delta ** 0.5) / (-2 * A)  # 计算解1：x1
#         x2 = (B - delta ** 0.5) / (-2 * A)  # 计算解2：x2
#         return x1, x2


def solve_quad_interval(A, B, C):
    """
    求解二次不等式 A*x^2 + B*x + C <= 0 的解区间。
    返回 (x1, x2) 表示区间 [x1, x2]，若无解返回 (None, None)
    """
    if A == 0:
        if B == 0:
            return (None, None) if C > 0 else (-float('inf'), float('inf'))
        root = -C / B
        return (-float('inf'), root) if B > 0 else (root, float('inf'))

    delta = B**2 - 4*A*C
    if delta < 0:
        return (None, None) if A > 0 else (-float('inf'), float('inf'))
    elif delta == 0:
        root = -B / (2*A)
        return (root, root)
    else:
        sqrt_d = delta ** 0.5
        x1 = (-B - sqrt_d) / (2*A)
        x2 = (-B + sqrt_d) / (2*A)
        if A > 0:
            return (min(x1, x2), max(x1, x2))
        else:
            # A < 0: inequality holds outside roots → not used in our variance case (A = w-1 ≥ 9)
            return (-float('inf'), min(x1, x2)), (max(x1, x2), float('inf'))


def speed_constraint_clean_local(dataframe: pd.DataFrame, speed_constraints, ra: list, t_attr, w=100):
    # Ensure sorted by time
    dataframe = dataframe.sort_values(t_attr).reset_index(drop=True)
    assert t_attr in dataframe.columns
    for attr in ra:
        assert attr in dataframe.columns
        assert attr in speed_constraints.keys()
    
    ra_range = [speed_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range])
    ra_vub = np.array([x[1] for x in ra_range])
    data = np.array(dataframe[ra], dtype=float)
    t = np.array(dataframe[t_attr], dtype=float)
    
    for i in range(1, dataframe.shape[0]):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue  # skip non-increasing timestamps
        x_min = data[i - 1] + ra_vlb * dt
        x_max = data[i - 1] + ra_vub * dt
        # Clamp current value to [x_min, x_max]
        data[i] = np.clip(data[i], x_min, x_max)
    
    dataframe[ra] = data


def speed_constraint_clean_global(dataframe: pd.DataFrame, speed_constraints, ra: list, t_attr, w=100,
                                  size=200, overlapping_ratio=0.2):
    dataframe = dataframe.sort_values(t_attr).reset_index(drop=True)
    assert t_attr in dataframe.columns
    for attr in ra:
        assert attr in dataframe.columns
        assert attr in speed_constraints.keys()
    
    ra_range = [speed_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range])
    ra_vub = np.array([x[1] for x in ra_range])
    
    for k in range(len(ra)):
        s = 0
        n_total = dataframe.shape[0]
        while s < n_total:
            e = min(s + size, n_total)
            data = np.array(dataframe.loc[s:e, ra], dtype=float)
            t = np.array(dataframe.loc[s:e, t_attr], dtype=float)
            n = data.shape[0]
            if n < 2:
                s += int((1 - overlapping_ratio) * size)
                continue

            c = np.ones(2 * n)
            A_ub = []
            b_ub = []
            bounds = [(0, None) for _ in range(2 * n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if t[j] > t[i] + w or t[j] <= t[i]:
                        break
                    dt = t[j] - t[i]
                    # Upper bound: x_j - x_i <= v_ub * dt
                    bij_max = ra_vub[k] * dt - (data[j, k] - data[i, k])
                    # Lower bound: x_j - x_i >= v_lb * dt  =>  -(x_j - x_i) <= -v_lb * dt
                    bij_min = -ra_vlb[k] * dt + (data[j, k] - data[i, k])

                    a_max = np.zeros(2 * n)
                    a_max[j] = 1
                    a_max[i] = -1
                    a_max[j + n] = -1
                    a_max[i + n] = 1
                    A_ub.append(a_max)
                    b_ub.append(bij_max)

                    a_min = np.zeros(2 * n)
                    a_min[j] = -1
                    a_min[i] = 1
                    a_min[j + n] = 1
                    a_min[i + n] = -1
                    A_ub.append(a_min)
                    b_ub.append(bij_min)

            if not A_ub:
                s += int((1 - overlapping_ratio) * size)
                continue

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                corrected = (res.x[:n] - res.x[n:]) + data[:, k]
                dataframe.loc[s:e, ra[k]] = corrected
            else:
                pass  # Optionally log failure; keep original

            s += int((1 - overlapping_ratio) * size)


def speed_plus_acceleration_constraint_clean_global(
    dataframe: pd.DataFrame, 
    speed_constraints, 
    acceleration_constraints,
    ra: list, 
    t_attr, 
    w=50, 
    size=100, 
    overlapping_ratio=0.1
):
    dataframe = dataframe.sort_values(t_attr).reset_index(drop=True)
    assert t_attr in dataframe.columns
    for attr in ra:
        assert attr in dataframe.columns
        assert attr in speed_constraints
        assert attr in acceleration_constraints

    ra_vlb = np.array([speed_constraints[attr][0] for attr in ra])
    ra_vub = np.array([speed_constraints[attr][1] for attr in ra])
    ra_alb = np.array([acceleration_constraints[attr][0] for attr in ra])
    ra_aub = np.array([acceleration_constraints[attr][1] for attr in ra])

    for k in range(len(ra)):
        s = 0
        n_total = dataframe.shape[0]
        while s < n_total:
            e = min(s + size, n_total)
            data = np.array(dataframe.loc[s:e, ra], dtype=float)
            t = np.array(dataframe.loc[s:e, t_attr], dtype=float)
            n = data.shape[0]
            if n < 3:
                s += int((1 - overlapping_ratio) * size)
                continue

            c = np.ones(2 * n)
            A_ub = []
            b_ub = []
            bounds = [(0, None) for _ in range(2 * n)]

            for i in range(n):
                for j in range(i + 1, n):
                    if t[j] > t[i] + w or t[j] <= t[i]:
                        break
                    dt_ij = t[j] - t[i]
                    # Speed constraints
                    bij_max_v = ra_vub[k] * dt_ij - (data[j, k] - data[i, k])
                    bij_min_v = -ra_vlb[k] * dt_ij + (data[j, k] - data[i, k])

                    a_max_v = np.zeros(2 * n)
                    a_max_v[j] = 1; a_max_v[i] = -1
                    a_max_v[j + n] = -1; a_max_v[i + n] = 1
                    A_ub.append(a_max_v); b_ub.append(bij_max_v)

                    a_min_v = np.zeros(2 * n)
                    a_min_v[j] = -1; a_min_v[i] = 1
                    a_min_v[j + n] = 1; a_min_v[i + n] = -1
                    A_ub.append(a_min_v); b_ub.append(bij_min_v)

                    # Acceleration constraints (requires i >= 1)
                    if i >= 1:
                        dt_i_prev = t[i] - t[i - 1]
                        if dt_ij <= 0 or dt_i_prev <= 0:
                            continue
                        # Current velocity estimate: v_i = (x_i - x_{i-1}) / dt_i_prev
                        # Future velocity: v_j = (x_j - x_i) / dt_ij
                        # Constraint: (v_j - v_i) / ((t_j - t_{i-1})/2) ≤ a_ub  → simplified as:
                        # (v_j - v_i) ≤ a_ub * dt_ij
                        # => (x_j - x_i)/dt_ij - (x_i - x_{i-1})/dt_i_prev ≤ a_ub * dt_ij
                        # Multiply both sides by dt_ij:
                        # (x_j - x_i) - (x_i - x_{i-1}) * (dt_ij / dt_i_prev) ≤ a_ub * dt_ij^2
                        scale = dt_ij / dt_i_prev
                        rhs_ub = ra_aub[k] * dt_ij * dt_ij + data[i, k] - data[i - 1, k] * scale
                        rhs_lb = -ra_alb[k] * dt_ij * dt_ij + data[i, k] - data[i - 1, k] * scale

                        a_ub_acc = np.zeros(2 * n)
                        a_ub_acc[j] = 1
                        a_ub_acc[i] = -1 - scale
                        a_ub_acc[i - 1] = scale
                        a_ub_acc[j + n] = -1
                        a_ub_acc[i + n] = 1 + scale
                        a_ub_acc[i - 1 + n] = -scale
                        A_ub.append(a_ub_acc)
                        b_ub.append(rhs_ub - (data[j, k] - data[i, k] - (data[i, k] - data[i - 1, k]) * scale))

                        a_lb_acc = np.zeros(2 * n)
                        a_lb_acc[j] = -1
                        a_lb_acc[i] = 1 + scale
                        a_lb_acc[i - 1] = -scale
                        a_lb_acc[j + n] = 1
                        a_lb_acc[i + n] = -1 - scale
                        a_lb_acc[i - 1 + n] = scale
                        A_ub.append(a_lb_acc)
                        b_ub.append(-rhs_lb + (data[j, k] - data[i, k] - (data[i, k] - data[i - 1, k]) * scale))

            if not A_ub:
                s += int((1 - overlapping_ratio) * size)
                continue

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                corrected = (res.x[:n] - res.x[n:]) + data[:, k]
                dataframe.loc[s:e, ra[k]] = corrected
            s += int((1 - overlapping_ratio) * size)


def variance_constraint_clean(dataframe: pd.DataFrame, variance_constraints: dict, ra: list, t_attr: str, w: int = 10, beta: float = 0.5):
    dataframe = dataframe.sort_values(t_attr).reset_index(drop=True)
    assert t_attr in dataframe.columns, f"时间列 '{t_attr}' 不存在"
    
    for attr in ra:
        assert attr in dataframe.columns, f"属性 '{attr}' 不存在"
        assert attr in variance_constraints, f"未提供 '{attr}' 的方差约束"
        assert isinstance(variance_constraints[attr], (tuple, list)) and len(variance_constraints[attr]) == 2, \
            f"variance_constraints['{attr}'] 应为 (min_var, max_var) 形式"
        
        min_var, max_var = variance_constraints[attr]
        assert 0 <= min_var <= max_var, f"方差约束应满足 0 <= min_var <= max_var，但得到 ({min_var}, {max_var})"
        
        sequence = np.array(dataframe[attr], dtype=float)
        n = len(sequence)
        if n < w:
            continue

        for k in range(w - 1, n - w + 1):
            repair_sum = 0.0
            weight_sum = 0.0
            
            for i in range(max(0, k - w + 1), min(k + 1, n - w + 1)):
                window = sequence[i:i+w]
                idx_in_window = k - i
                if idx_in_window < 0 or idx_in_window >= w:
                    continue
                current_var = np.var(window)
                num = k - i
                weight = beta ** num
                weight_sum += weight

                if min_var <= current_var <= max_var:
                    repair_sum += weight * sequence[k]
                    continue

                others = np.delete(window, idx_in_window)
                sum_others = np.sum(others)
                sum2_others = np.sum(others * others)
                A = w - 1
                B = -2 * sum_others
                C_low = (w * sum2_others - sum_others**2) - max_var * w**2
                C_high = (w * sum2_others - sum_others**2) - min_var * w**2

                x1_low, x2_low = solve_quad_interval(A, B, C_low)
                x1_high, x2_high = solve_quad_interval(A, B, C_high)

                feasible_intervals = []
                if x1_low is not None and x2_low is not None:
                    L1, R1 = x1_low, x2_low
                    if x1_high is not None and x2_high is not None:
                        L2, R2 = x1_high, x2_high
                        # Intersection: [L1,R1] ∩ ((-∞,L2] ∪ [R2,∞))
                        if L1 <= L2:
                            end = min(R1, L2)
                            if L1 <= end:
                                feasible_intervals.append((L1, end))
                        if R2 <= R1:
                            start = max(L1, R2)
                            if start <= R1:
                                feasible_intervals.append((start, R1))
                    else:
                        feasible_intervals.append((L1, R1))

                if not feasible_intervals:
                    repair_sum += weight * sequence[k]  # fallback
                    continue

                best_x = sequence[k]
                min_dist = float('inf')
                for a, b in feasible_intervals:
                    if a <= sequence[k] <= b:
                        best_x = sequence[k]
                        break
                    for cand in [a, b]:
                        dist = abs(cand - sequence[k])
                        if dist < min_dist:
                            min_dist = dist
                            best_x = cand
                repair_sum += weight * best_x

            if weight_sum > 0:
                repair_val = repair_sum / weight_sum
                if abs(repair_val - sequence[k]) > 1e-6:
                    sequence[k] = repair_val

        dataframe[attr] = sequence


def speed_plus_acceleration_constraint_clean_local(dataframe: pd.DataFrame, speed_constraints, acceleration_constraints,
                                                   ra: list, t_attr, w=100):
    dataframe = dataframe.sort_values(t_attr).reset_index(drop=True)
    assert t_attr in dataframe.columns
    for attr in ra:
        assert attr in dataframe.columns
        assert attr in speed_constraints
        assert attr in acceleration_constraints

    ra_vlb = np.array([speed_constraints[attr][0] for attr in ra])
    ra_vub = np.array([speed_constraints[attr][1] for attr in ra])
    ra_alb = np.array([acceleration_constraints[attr][0] for attr in ra])
    ra_aub = np.array([acceleration_constraints[attr][1] for attr in ra])

    data = np.array(dataframe[ra], dtype=float)
    t = np.array(dataframe[t_attr], dtype=float)

    for k in range(2, dataframe.shape[0]):
        dt1 = t[k] - t[k-1]
        dt2 = t[k-1] - t[k-2]
        if dt1 <= 0 or dt2 <= 0:
            continue

        # Speed-based bounds
        x_min_v = data[k-1] + ra_vlb * dt1
        x_max_v = data[k-1] + ra_vub * dt1

        # Acceleration-based bounds
        v_prev = (data[k-1] - data[k-2]) / dt2
        x_min_a = data[k-1] + (v_prev + ra_alb) * dt1
        x_max_a = data[k-1] + (v_prev + ra_aub) * dt1

        x_min = np.maximum(x_min_v, x_min_a)
        x_max = np.minimum(x_max_v, x_max_a)

        data[k] = np.clip(data[k], x_min, x_max)

    dataframe[ra] = data




if __name__ == '__main__':
    # =======  构造脏数据 ==========
    np.random.seed(0)
    N = 200

    df = pd.DataFrame({
        "timestamp": np.arange(N),
        "temp": np.sin(np.linspace(0, 6, N)) * 30 + np.random.randn(N) * 0.5,
        "pressure": np.cos(np.linspace(0, 5, N)) * 50 + np.random.randn(N) * 1,
    })

    # 制造速度违规（跳变点）
    df.loc[50, "temp"] = 200
    df.loc[120, "pressure"] = 300

    # 制造方差异常
    df.loc[30:35, "temp"] += 50

    print("原始数据：")
    print(df.head(10))

    # =======  设置约束 ==========
    speed_constraints = {
        "temp": ( -2,  2),     # 最小 / 最大速度
        "pressure": (-3, 3)
    }
    acc_constraints = {
        "temp": (-0.5, 0.5),   # 最小 / 最大加速度
        "pressure": (-0.7, 0.7)
    }
    variance_constraints = {
        "temp": (0,100),
        "pressure": (0,120)
    }

    # # =========  调用本地速度约束修复 ==========
    # df1 = df.copy()
    # speed_constraint_clean_local(df1, speed_constraints, ["temp","pressure"], "timestamp", w=10)

    # print("\n本地速度约束清洗后(前10行)：")
    # print(df1.head(10))

    # # =========  调用全局速度约束修复 ==========
    # df2 = df.copy()
    # speed_constraint_clean_global(df2, speed_constraints, ["temp","pressure"], "timestamp")
    # print("\n全局速度约束清洗后(前10行)：")
    # print(df2.head(10))

    # # =========  调用速度+加速度本地修复 ==========
    # df3 = df.copy()
    # speed_plus_acceleration_constraint_clean_local(df3, speed_constraints, acc_constraints, ["temp","pressure"], "timestamp")
    # print("\n速度+加速度本地(前10行)：")
    # print(df3.head(10))

    # =========  调用速度+加速度全局修复 ==========
    df4 = df.copy()
    speed_plus_acceleration_constraint_clean_global(df4, speed_constraints, acc_constraints, ["temp","pressure"], "timestamp")
    print("\n速度+加速度全局(前10行)：")
    print(df4.head(10))

    # # =========  方差约束修复 ==========
    # df5 = df.copy()
    # variance_constraint_clean(df5, variance_constraints, ["temp"], "timestamp", w=10)
    # print("\n方差约束清洗后(前10行)：")
    # print(df5.head(10))

