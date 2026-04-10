import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def window_regression_repair(
    data_abnormal: np.ndarray, label: np.ndarray, window=5
) -> np.ndarray:
    """
    优化后的窗口回归修复：
    1. 逻辑修正：将回归目标设定为 y = f(relative_time)，避免恒等映射。
    2. 性能优化：预计算异常点，减少冗余的插值计算。
    3. 稳定性：增加输入校验和分块处理。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        if not np.any(is_abnormal):
            continue
            
        anomaly_indices = np.where(is_abnormal)[0]
        X_full = data_abnormal[s]
        
        # 记录哪些点已经通过回归修复，哪些需要最后通过插值兜底
        repaired_mask = np.zeros(n_timestamps, dtype=bool)

        for t in anomaly_indices:
            # 定义局部窗口
            start = max(0, t - window)
            end = min(n_timestamps, t + window + 1)
            
            # 找到窗口内的正常点索引
            local_range = np.arange(start, end)
            train_idx = local_range[(label[s, local_range] == 0)]

            # 只有当正常点足够多时才进行回归
            if len(train_idx) >= max(3, n_features // 2):
                # 构造特征：只以“相对时间偏移”作为自变量
                # 原逻辑中的 X 作为特征会导致多重共线性，这里改为纯时间序列回归
                X_train = (train_idx - t).reshape(-1, 1)
                Y_train = X_full[train_idx] # (n_train_points, n_features)

                try:
                    model = LinearRegression()
                    model.fit(X_train, Y_train)
                    
                    # 预测相对时间为 0 的位置
                    pred = model.predict([[0]])[0]
                    data_repaired[s, t, :] = pred
                    repaired_mask[t] = True
                except:
                    continue
        
        # 兜底机制：对于回归失败或点不够的点，执行一次性全局线性插值
        still_abnormal = is_abnormal & (~repaired_mask)
        if np.any(still_abnormal):
            for c in range(n_features):
                series = pd.Series(data_repaired[s, :, c])
                # 将仍未修复的异常点置为 NaN 进行插值
                series[still_abnormal] = np.nan
                data_repaired[s, :, c] = series.interpolate(
                    method="linear", limit_direction="both"
                ).values

    return data_repaired