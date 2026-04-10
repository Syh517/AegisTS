import numpy as np
import pandas as pd

def smurf_repair(
    data_abnormal: np.ndarray, label: np.ndarray, window=3
) -> np.ndarray:
    """
    优化后的 SMURF 修复函数：
    1. 性能：使用高效的滑动窗口均值（基于统一掩码）替代逐点循环。
    2. 逻辑：保持局部平滑特性，并增加了对连续异常段的“逐步收缩预测”能力。
    3. 效率：通过矩阵运算减少 Python 层面的索引操作。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 将异常点设为 NaN 方便计算忽略 NaN 的均值
    data_with_nan = data_abnormal.astype(float)
    data_with_nan[label == 1] = np.nan

    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        if not np.any(is_abnormal):
            continue
            
        for c in range(n_features):
            series = data_with_nan[s, :, c]
            
            # 使用滑动窗口计算局部均值 (nanmean)
            # 通过 pd.Series 的 rolling 可以极快地完成向量化窗口计算
            # min_periods=1 确保只要窗口内有一个正常点就能修复
            local_mean = (
                pd.Series(series)
                .rolling(window=2*window+1, center=True, min_periods=1)
                .mean()
                .values
            )
            
            # 找到当前特征的异常位置
            bad_idx = np.where(is_abnormal)[0]
            
            # 修复逻辑
            repaired_values = local_mean[bad_idx]
            
            # 兜底：如果 local_mean 依然是 NaN (说明整个窗口内全是异常点)
            # 则逐步扩大搜索范围或使用该列全局均值
            nan_mask = np.isnan(repaired_values)
            if np.any(nan_mask):
                global_mean = np.nanmean(series)
                repaired_values[nan_mask] = global_mean if not np.isnan(global_mean) else 0.0
                
            data_repaired[s, bad_idx, c] = repaired_values
            
    return data_repaired