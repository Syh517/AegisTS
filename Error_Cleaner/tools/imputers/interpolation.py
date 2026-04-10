import numpy as np
import warnings
from scipy.interpolate import interp1d

def impute_interpolation(data_missing: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    优化后的插值法：高性能、抗振荡、全流程有限值保障。
    
    :param data_missing: 待修复的 numpy 数组 (N x T x D)
    :param method: 插值方法 ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
    :return: 修复后的 numpy 数组 (N x T x D)
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    # 1. 初始化与预处理
    data_imputed = np.copy(data_missing)
    # 将 inf 转为 nan
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    x_indices = np.arange(n_timesteps)

    # 2. 全局特征均值兜底
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本逐特征处理 (纯 NumPy 逻辑)
    for i in range(n_samples):
        for d in range(n_features):
            series = data_imputed[i, :, d]
            nan_mask = np.isnan(series)
            
            if not np.any(nan_mask):
                continue
            
            valid_idx = np.where(~nan_mask)[0]
            
            # 情况 A: 整列缺失
            if valid_idx.size == 0:
                data_imputed[i, :, d] = global_feature_means[d]
                continue
            
            # 记录有效值的边界，用于防止插值溢出
            s_min, s_max = np.min(series[valid_idx]), np.max(series[valid_idx])
            
            # 情况 B: 只有一个点，无法插值，直接填充
            if valid_idx.size == 1:
                data_imputed[i, :, d] = series[valid_idx[0]]
                continue

            # 情况 C: 正常插值
            try:
                # 确定当前的插值方法：如果点数不足以进行高阶插值，降级为 linear
                current_method = method
                if method in ['quadratic', 'cubic'] and valid_idx.size < 4:
                    current_method = 'linear'
                
                # 创建插值函数 (bounds_error=False 允许外推或配合 fill_value)
                # 使用 fill_value="extrapolate" 或自定义边界处理
                f = interp1d(valid_idx, series[valid_idx], 
                             kind=current_method, 
                             fill_value="extrapolate", 
                             assume_sorted=True)
                
                interp_values = f(x_indices)
                
                # 安全性：数值截断 (Clipping)
                # 高阶插值极易在边缘产生巨大的波峰或波谷，限制其范围
                interp_values = np.clip(interp_values, s_min, s_max)
                
                # 仅覆盖缺失部分
                series[nan_mask] = interp_values[nan_mask]
                
            except Exception:
                # 如果 Scipy 插值失败，使用最稳健的 np.interp (仅限线性)
                series[nan_mask] = np.interp(x_indices[nan_mask], valid_idx, series[valid_idx])

            # 情况 D: 边缘兜底（处理 extrapolate 可能产生的 NaN/Inf）
            bad_mask = ~np.isfinite(series)
            if np.any(bad_mask):
                series[bad_mask] = global_feature_means[d]
            
            data_imputed[i, :, d] = series

    # 4. 最终全数组加固
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed