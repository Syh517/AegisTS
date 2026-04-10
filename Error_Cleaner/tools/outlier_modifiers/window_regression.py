import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def window_regression_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, window=10, sigma_threshold=3
) -> np.ndarray:
    """
    自适应窗口回归修复函数：
    1. 自动识别：在滑动窗口内计算残差，利用 3-Sigma 原则自动锁定脏数据。
    2. 局部回归：以时间偏移量为自变量进行线性回归，捕捉局部趋势。
    3. 动态修复：只修复那些显著偏离局部回归线的点或原始 NaN 点。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for s in range(n_samples):
        # 记录该样本哪些点已被修复（用于后续兜底判断）
        repaired_mask = np.zeros(n_timestamps, dtype=bool)
        # 预识别 NaN 点，这些点必须修复
        must_repair = np.isnan(data_abnormal[s]).any(axis=1)

        for t in range(n_timestamps):
            # 定义局部窗口范围
            start = max(0, t - window)
            end = min(n_timestamps, t + window + 1)
            local_indices = np.arange(start, end)
            
            # 1. 筛选窗口内的候选正常点（非 NaN）
            # 注意：在没有 label 的情况下，我们初筛掉明显的 NaN
            valid_mask = ~np.isnan(data_abnormal[s, local_indices]).any(axis=1)
            train_idx = local_indices[valid_mask]

            # 只有当窗口内有效点足够时才进行建模
            if len(train_idx) >= 5:
                # 构造特征：时间偏移
                X_train = (train_idx - t).reshape(-1, 1)
                Y_train = data_abnormal[s, train_idx]

                try:
                    model = LinearRegression()
                    model.fit(X_train, Y_train)
                    
                    # 预测当前点 t 的值 (相对偏移为 0)
                    pred = model.predict([[0]])[0]
                    
                    # 2. 自动判定脏数据逻辑：
                    # 如果 t 本身是 NaN，直接修复
                    # 如果 t 不是 NaN，计算其观测值与预测值的残差
                    if must_repair[t]:
                        data_repaired[s, t, :] = pred
                        repaired_mask[t] = True
                    else:
                        actual = data_abnormal[s, t, :]
                        # 计算窗口内所有正常点的平均残差标准差
                        train_preds = model.predict(X_train)
                        resid_std = np.std(Y_train - train_preds, axis=0)
                        
                        # 判定：如果观测值偏离预测值超过 sigma_threshold 倍标准差，视为脏数据
                        # 只要有一个特征维度超标，就进行全维度修复（保持变量间协同）
                        if np.any(np.abs(actual - pred) > sigma_threshold * resid_std):
                            data_repaired[s, t, :] = pred
                            repaired_mask[t] = True
                except:
                    continue

        # --- Step 3: 兜底机制 ---
        # 对于回归覆盖不到的区域（如开头结尾或连续缺失严重的区域），使用线性插值
        still_unfixed = must_repair & (~repaired_mask)
        if np.any(still_unfixed):
            for c in range(n_features):
                series = pd.Series(data_repaired[s, :, c])
                # 将依然脏/缺的点设为 NaN
                series[still_unfixed] = np.nan
                # 使用线性插值补全
                data_repaired[s, :, c] = series.interpolate(
                    method="linear", limit_direction="both"
                ).values

    return data_repaired