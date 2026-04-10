import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def exponential_smoothing_repair(
    data_abnormal: np.ndarray, label: np.ndarray, alpha=0.5
) -> np.ndarray:
    """
    使用指数平滑方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        alpha: 平滑参数
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            # 剔除异常点
            series[~mask] = np.nan
            # 插值补全用于拟合
            series_pd = pd.Series(series)
            series_filled = (
                series_pd.interpolate().fillna(method="bfill").fillna(method="ffill")
            ).values

            model = ExponentialSmoothing(series_filled, trend=None, seasonal=None)
            fitted = model.fit(smoothing_level=alpha, optimized=False)
            preds = fitted.fittedvalues
            data_repaired[sample, label[sample] == 1, col] = preds[label[sample] == 1]
        
    return data_repaired