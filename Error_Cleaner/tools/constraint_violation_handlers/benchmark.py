import numpy as np
import pandas as pd
from scipy.optimize import linprog


def speed_constraint_clean_local(dataframe, speed_constraints, ra: list, t_attr, w=100):
    """
    匹配接口调用：直接接收 dataframe
    speed_constraints 格式: {'col_1': (min, max), ...}
    """
    # 提取约束
    # 注意：只处理在 ra 列表且在 constraints 字典中都存在的列
    valid_ra = [attr for attr in ra if attr in speed_constraints]
    if not valid_ra:
        return

    ra_vlb = np.array([speed_constraints[attr][0] for attr in valid_ra])
    ra_vub = np.array([speed_constraints[attr][1] for attr in valid_ra])
    
    # 转换为 numpy 提高计算效率
    data = dataframe[valid_ra].to_numpy(dtype=np.float64)
    t = dataframe[t_attr].to_numpy(dtype=np.float64)
    
    for i in range(1, len(data) - 1):
        dt_prev = t[i] - t[i - 1]
        x_i_min = ra_vlb * dt_prev + data[i - 1]
        x_i_max = ra_vub * dt_prev + data[i - 1]
        
        candidate_i = [data[i]]
        for k in range(i + 1, len(data)):
            dt_k = t[k] - t[i]
            if dt_k > w:
                break
            # 候选值推导
            candidate_i.append(data[k] - ra_vlb * dt_k)
            candidate_i.append(data[k] - ra_vub * dt_k)
        
        x_i_mid = np.median(np.array(candidate_i), axis=0)
        # 裁剪到速度约束范围内
        data[i] = np.clip(x_i_mid, x_i_min, x_i_max)
    
    # 写回原 DataFrame
    dataframe[valid_ra] = data


def speed_constraint_clean_global(dataframe, speed_constraints, ra: list, t_attr, w=100,
                                 size=200, overlapping_ratio=0.2):
    """
    匹配接口调用：直接接收 dataframe，执行线性规划优化
    """
    for attr in ra:
        if attr not in speed_constraints:
            continue
            
        vlb, vub = speed_constraints[attr]
        s = 0
        n_total = len(dataframe)
        
        while s < n_total:
            e = min(s + size, n_total)
            # 使用 iloc 确保索引连续性
            subset = dataframe.iloc[s:e]
            data = subset[attr].to_numpy(dtype=np.float64)
            t = subset[t_attr].to_numpy(dtype=np.float64)
            n = len(data)
            
            # 构建线性规划：minimize sum(e_plus + e_minus)
            c = np.ones(2 * n)
            A_ub = []
            b_ub = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    dt = t[j] - t[i]
                    if dt > w: break
                    
                    diff_orig = data[j] - data[i]
                    # 上界约束: (ej - ei) <= vub*dt - (dataj - datai)
                    row_max = np.zeros(2 * n)
                    row_max[j], row_max[j+n], row_max[i], row_max[i+n] = 1, -1, -1, 1
                    A_ub.append(row_max)
                    b_ub.append(vub * dt - diff_orig)
                    
                    # 下界约束: (ei - ej) <= -vlb*dt + (dataj - datai)
                    row_min = np.zeros(2 * n)
                    row_min[i], row_min[i+n], row_min[j], row_min[j+n] = 1, -1, -1, 1
                    A_ub.append(row_min)
                    b_ub.append(-vlb * dt + diff_orig)
            
            if A_ub:
                res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                              bounds=[(0, None)] * (2 * n), method='highs')
                if res.success:
                    # 更新值 = 原始值 + (e_plus - e_minus)
                    corrections = res.x[:n] - res.x[n:]
                    dataframe.iloc[s:e, dataframe.columns.get_loc(attr)] = data + corrections
            
            s += int((1 - overlapping_ratio) * size)
            if s >= n_total - 1: break


def speed_plus_acceleration_constraint_clean_local(dataframe, speed_constraints, acceleration_constraints,
                                                  ra: list, t_attr, w=100):
    # 筛选出在 constraints 中有定义且在 DataFrame 中存在的列
    valid_ra = [attr for attr in ra if attr in speed_constraints and attr in acceleration_constraints]
    if not valid_ra:
        return

    ra_vlb = np.array([speed_constraints[attr][0] for attr in valid_ra])
    ra_vub = np.array([speed_constraints[attr][1] for attr in valid_ra])
    ra_alb = np.array([acceleration_constraints[attr][0] for attr in valid_ra])
    ra_aub = np.array([acceleration_constraints[attr][1] for attr in valid_ra])

    data = dataframe[valid_ra].to_numpy(dtype=np.float64)
    t = dataframe[t_attr].to_numpy(dtype=np.float64)
    
    # 局部修复从索引 2 开始（因为加速度需要前两个点计算初始速度）
    for k in range(2, len(data) - 1):
        dt = t[k] - t[k-1]
        dt_prev = t[k-1] - t[k-2]
        
        # 1. 速度限制下的 X_k 范围
        x_k_min_v = ra_vlb * dt + data[k-1]
        x_k_max_v = ra_vub * dt + data[k-1]
        
        # 2. 加速度限制下的 X_k 范围
        v_prev = (data[k-1] - data[k-2]) / dt_prev
        x_k_min_a = (ra_alb * dt + v_prev) * dt + data[k-1]
        x_k_max_a = (ra_aub * dt + v_prev) * dt + data[k-1]
        
        # 取交集
        x_k_min = np.maximum(x_k_min_v, x_k_min_a)
        x_k_max = np.minimum(x_k_max_v, x_k_max_a)
        
        # 窗口内候选值收集
        X_candidates = [data[k]]
        for i in range(k + 1, len(data)):
            if t[i] > t[k] + w:
                break
            dt_i = t[i] - t[k]
            # 基于未来点反推可能的值
            X_candidates.append(data[i] - ra_vlb * dt_i)
            X_candidates.append(data[i] - ra_vub * dt_i)
            
        x_k_mid = np.median(np.array(X_candidates), axis=0)
        # 最终裁剪
        data[k] = np.clip(x_k_mid, x_k_min, x_k_max)
        
    dataframe[valid_ra] = data


def speed_plus_acceleration_constraint_clean_global(dataframe, speed_constraints, acceleration_constraints,
                                                   ra: list, t_attr, w=50, size=100, overlapping_ratio=0.1):
    for attr in ra:
        if attr not in speed_constraints or attr not in acceleration_constraints:
            continue
            
        vlb, vub = speed_constraints[attr]
        alb, aub = acceleration_constraints[attr]
        s = 0
        n_total = len(dataframe)
        
        while s < n_total:
            e = min(s + size, n_total)
            # 使用 iloc 提取局部窗口
            data_win = dataframe[attr].iloc[s:e].to_numpy(dtype=np.float64)
            t_win = dataframe[t_attr].iloc[s:e].to_numpy(dtype=np.float64)
            n = len(data_win)
            if n < 3: break # 加速度约束至少需要3个点
            
            c = np.ones(2 * n) # 目标函数变量: [e_plus, e_minus]
            A_ub = []
            b_ub = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    dt = t_win[j] - t_win[i]
                    if dt > w: break
                    diff_orig = data_win[j] - data_win[i]
                    
                    # 速度约束矩阵行构建
                    row_v_max = np.zeros(2 * n)
                    row_v_max[j], row_v_max[j+n], row_v_max[i], row_v_max[i+n] = 1, -1, -1, 1
                    A_ub.append(row_v_max)
                    b_ub.append(vub * dt - diff_orig)
                    
                    row_v_min = np.zeros(2 * n)
                    row_v_min[i], row_v_min[i+n], row_v_min[j], row_v_min[j+n] = 1, -1, -1, 1
                    A_ub.append(row_v_min)
                    b_ub.append(-vlb * dt + diff_orig)
                    
                    # 加速度约束
                    if i >= 1:
                        dt_p = t_win[i] - t_win[i-1]
                        # 差分近似加速度公式: ( (Xj-Xi)/dt - (Xi-Xi-1)/dt_p ) / dt_avg
                        # 简化为: (Xj-Xi)/dt - (Xi-Xi-1)/dt_p <= aub * dt
                        tmp1, tmp2 = 1.0/dt, 1.0/dt_p
                        accel_orig = (data_win[j]-data_win[i])*tmp1 - (data_win[i]-data_win[i-1])*tmp2
                        
                        row_a_max = np.zeros(2 * n)
                        row_a_max[j], row_a_max[j+n] = tmp1, -tmp1
                        row_a_max[i], row_a_max[i+n] = -tmp1 - tmp2, tmp1 + tmp2
                        row_a_max[i-1], row_a_max[i-1+n] = tmp2, -tmp2
                        A_ub.append(row_a_max)
                        b_ub.append(aub * dt - accel_orig)
            
            if A_ub:
                res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                              bounds=[(0, None)] * (2 * n), method='highs')
                if res.success:
                    corrections = res.x[:n] - res.x[n:]
                    dataframe.iloc[s:e, dataframe.columns.get_loc(attr)] = data_win + corrections
            
            s += int((1 - overlapping_ratio) * size)
            if s >= n_total - 2: break

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

def variance_constraint_clean(dataframe: pd.DataFrame, variance_constraints: dict, ra: list, t_attr, w=10, beta=0.5):
    """
    匹配接口调用：直接处理传入的 dataframe
    variance_constraints 格式: {'col_1': (min_v, max_v), ...}
    """
    for attr in ra:
        # 仅处理存在于约束字典中的列
        if attr not in dataframe.columns or attr not in variance_constraints:
            continue
            
        # 提取方差上限 (使用元组的第二个值)
        max_var = variance_constraints[attr][1]
        
        sequence = dataframe[attr].to_numpy(dtype=np.float64)
        n = len(sequence)
        
        # 遍历序列进行修复 (范围需确保滑动窗口不越界)
        for k in range(w - 1, n - w):
            repair_sum = 0
            weight_sum = 0
            
            # 考量所有包含点 k 的窗口 (窗口起始点从 k-w+1 到 k)
            for i in range(k - w + 1, k + 1):
                if i < 0: continue
                
                num = i - (k - w + 1)
                weight = pow(beta, num)
                weight_sum += weight
                
                # 计算当前窗口的方差
                window = sequence[i : i + w]
                if np.var(window) <= max_var:
                    repair_sum += weight * sequence[k]
                else:
                    # 方差超标，通过二次方程求解理想的修复值 x
                    # 令窗口内除 k 点外的元素之和为 l1，平方和为 l2
                    l1 = np.sum(window) - sequence[k]
                    l2 = np.sum(window**2) - sequence[k]**2
                    
                    # 构建方程: (w-1)x^2 - (2*l1)x + (w*l2 - l1^2 - w^2*max_var) = 0
                    a_coeff = w - 1
                    b_coeff = -2 * l1
                    c_coeff = w * l2 - l1**2 - (w**2 * max_var)
                    
                    x1, x2 = solve_quad_interval(a_coeff, b_coeff, c_coeff)
                    
                    # 选择最接近原值的根
                    if x1 is None:
                        repair_sum += weight * sequence[k]
                    elif abs(x1 - sequence[k]) < abs(x2 - sequence[k]):
                        repair_sum += weight * x1
                    else:
                        repair_sum += weight * x2
            
            # 执行加权平均修复
            if weight_sum > 0:
                repair_val = repair_sum / weight_sum
                sequence[k] = repair_val
        
        # 将修复后的数组写回 DataFrame
        dataframe[attr] = sequence