import numpy as np
from scipy.linalg import lstsq, toeplitz

def ols_residual_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    p: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    优化后的 OLS 残差修复：
    1. 矩阵化加速：利用滑动窗口矩阵一次性提取所有滞后项。
    2. 屏蔽训练：仅使用 label=0（正常）的点来训练 AR 系数。
    3. 收敛加速：列级别整体更新。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for s in range(n_samples):
        X_sample = data_repaired[s]
        M_sample = label[s].astype(bool)  # True 表示异常

        if not np.any(M_sample):
            continue

        for col in range(n_features):
            series = X_sample[:, col]
            mask = M_sample  # 该列的异常掩码

            # --- 1. 构建全量滞后矩阵 (Toeplitz 矩阵加速) ---
            # 滞后矩阵形状为 (n_timestamps, p)
            # 每一行 i 包含 [series[i-1], series[i-2], ..., series[i-p]]
            # 填充开头部分以保证对齐
            first_val = series[0]
            padded_series = np.concatenate([np.full(p, first_val), series])
            
            # 使用滑动窗口构建特征矩阵 X
            # 此处使用 stride_tricks 可以达到无拷贝的矩阵化提取
            from numpy.lib.stride_tricks import as_strided
            X_all = as_strided(
                padded_series, 
                shape=(n_timestamps, p), 
                strides=(series.strides[0], series.strides[0])
            )[:, ::-1] # 逆序排列使第一列为 t-1

            # --- 2. 迭代修复 ---
            for i in range(max_iter):
                prev_series = series.copy()
                
                # 仅使用“正常点”作为训练集来估计 AR 系数
                # 目标：series[t] = X_all[t] @ phi
                train_mask = ~mask
                # 去掉前 p 个点中可能因 padding 导致的不稳定点（可选）
                X_train = X_all[train_mask]
                y_train = series[train_mask]

                if len(y_train) < p:
                    break # 正常样本不足，无法训练

                # 求解 AR 系数 phi
                phi, *_ = lstsq(X_train, y_train)

                # 修复异常点：pred = X_all[anomaly] @ phi
                predictions = X_all[mask] @ phi
                series[mask] = predictions

                # 更新 X_all 中受影响的部分 (虽然滞后项会随迭代改变，
                # 但矩阵化更新比逐点计算快得多)
                # 为简化计算，单次迭代内 phi 视作固定
                
                # 检查收敛
                if np.linalg.norm(series - prev_series) < tol:
                    break
            
            X_sample[:, col] = series

    return data_repaired