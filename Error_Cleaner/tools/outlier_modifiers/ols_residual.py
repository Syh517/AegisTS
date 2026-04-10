import numpy as np
from scipy.linalg import lstsq
from numpy.lib.stride_tricks import as_strided

def ols_residual_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None,
    p: int = 10,
    max_iter: int = 50,
    tol: float = 1e-6,
    threshold_sigma: float = 3.0
) -> np.ndarray:
    """
    无监督 OLS 残差修复：
    1. 不依赖外部 label，通过残差自动识别异常点。
    2. 使用 3-Sigma 原则动态更新异常掩码。
    3. 迭代优化直到残差分布稳定。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for s in range(n_samples):
        X_sample = data_repaired[s]

        for col in range(n_features):
            series = X_sample[:, col]
            
            # 1. 处理初始无效值 (NaN/Inf)
            # 即使没有 label，我们也必须先处理掉物理上的空值
            finite_mask = np.isfinite(series)
            if not np.any(finite_mask):
                continue
            
            # 线性插值填充初始 NaN，为构建滞后矩阵做准备
            valid_indices = np.where(finite_mask)[0]
            series = np.interp(np.arange(len(series)), valid_indices, series[valid_indices])
            
            # 构建初始滞后矩阵 X_all
            first_val = series[0]
            padded = np.concatenate([np.full(p, first_val), series])
            X_all = as_strided(
                padded, 
                shape=(n_timestamps, p), 
                strides=(series.strides[0], series.strides[0])
            )[:, ::-1].copy()

            # 初始：假设所有点都是正常的
            current_anomaly_mask = np.zeros(n_timestamps, dtype=bool)

            # 2. 迭代：筛选异常点 -> 训练模型 -> 修复 -> 重新筛选
            for i in range(max_iter):
                prev_series = series.copy()
                
                # 训练：只使用当前判定为“非异常”的点
                train_mask = ~current_anomaly_mask
                X_train, y_train = X_all[train_mask], series[train_mask]

                if len(y_train) < p * 2: # 确保样本量足够
                    break

                try:
                    phi, *_ = lstsq(X_train, y_train)
                except:
                    break

                # 计算全量预测残差
                full_pred = X_all @ phi
                residuals = np.abs(series - full_pred)
                
                # 动态识别异常：残差大于 3 倍标准差的点定义为异常
                # 或者使用中位数绝对偏差 (MAD) 更加鲁棒
                std_res = np.std(residuals)
                new_anomaly_mask = residuals > (threshold_sigma * std_res)
                
                # 修复判定为异常的点
                series[new_anomaly_mask] = full_pred[new_anomaly_mask]

                # 更新 X_all 矩阵
                for lag in range(1, p + 1):
                    X_all[lag:, lag-1] = series[:-lag]
                    X_all[:lag, lag-1] = series[0]

                # 检查收敛：如果异常掩码不再变化或数值稳定
                if np.array_equal(new_anomaly_mask, current_anomaly_mask) or \
                   np.linalg.norm(series - prev_series) < tol:
                    break
                
                current_anomaly_mask = new_anomaly_mask
            
            X_sample[:, col] = series

    return data_repaired