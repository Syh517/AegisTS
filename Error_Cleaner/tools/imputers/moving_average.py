import numpy as np
import warnings
from scipy.ndimage import uniform_filter1d

def impute_moving_average(data_missing: np.ndarray, window: int = 5) -> np.ndarray:
    """
    优化后的滑动平均插补：利用向量化卷积加速，并确保无 NaN/Inf。
    
    :param data_missing: 待修复的 numpy 数组 (N x T x D)
    :param window: 滑动窗口大小
    :return: 修复后的 numpy 数组 (N x T x D)
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    # 1. 初始化与预处理
    data_imputed = np.copy(data_missing)
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 2. 全局特征均值兜底方案
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本、逐特征处理
    for i in range(n_samples):
        for d in range(n_features):
            series = data_imputed[i, :, d]
            nan_mask = np.isnan(series)
            
            if not np.any(nan_mask):
                continue
            
            # 情况 A: 整列全缺失
            valid_idx = np.where(~nan_mask)[0]
            if valid_idx.size == 0:
                data_imputed[i, :, d] = global_feature_means[d]
                continue

            # 记录有效值范围，用于安全性截断
            s_min, s_max = np.min(series[valid_idx]), np.max(series[valid_idx])
            
            # 情况 B: 部分缺失，使用向量化滑动平均
            # 先用线性插值填充一份基础数据，用于计算滑动窗口均值（避免窗口内全是 NaN）
            base_filled = np.interp(np.arange(n_timesteps), valid_idx, series[valid_idx])
            
            # 使用 uniform_filter1d 实现快速滑动平均 (等同于 pandas.rolling.mean)
            # mode='nearest' 处理边界
            ma_values = uniform_filter1d(base_filled, size=window, mode='nearest')
            
            # 安全性：截断 MA 结果，防止平滑后出现意外数值
            ma_values = np.clip(ma_values, s_min, s_max)
            
            # 仅在原始缺失位置填充 MA 结果
            series_filled = np.copy(series)
            series_filled[nan_mask] = ma_values[nan_mask]
            
            # 情况 C: 最终检查 (处理极端情况，如 MA 产生非有限值)
            final_bad_mask = ~np.isfinite(series_filled)
            if np.any(final_bad_mask):
                # 如果仍有异常，使用线性插值补齐
                series_filled[final_bad_mask] = base_filled[final_bad_mask]
                # 若线性插值还不行（极罕见），用全局均值
                still_bad = ~np.isfinite(series_filled)
                series_filled[still_bad] = global_feature_means[d]
                
            data_imputed[i, :, d] = series_filled

    # 4. 全局数值加固
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed