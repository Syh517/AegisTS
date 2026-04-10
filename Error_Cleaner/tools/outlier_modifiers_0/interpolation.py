import numpy as np
from scipy import interpolate

def interpolation_repair(
    data_abnormal: np.ndarray, label: np.ndarray, method="linear"
) -> np.ndarray:
    """
    优化后的插值修复函数
    
    参数:
        data_abnormal: (n_samples, n_timestamps, n_features)
        label: (n_samples, n_timestamps), 1为异常, 0为正常
        method: 插值方法, 如 'linear', 'quadratic', 'cubic', 'nearest'
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 预生成时间索引，避免在循环中重复生成
    x_full = np.arange(n_timestamps)
    
    for i in range(n_samples):
        # 提取当前样本的掩码
        is_abnormal = (label[i] == 1)
        if not np.any(is_abnormal):
            continue
            
        is_normal = ~is_abnormal
        x_good = x_full[is_normal]
        x_bad = x_full[is_abnormal]
        
        # 健壮性检查：如果正常点少于 2 个，插值无法进行
        if len(x_good) < 2:
            # 回退策略：使用全局均值或保持原样
            avg_vals = np.nanmean(data_abnormal[i], axis=0)
            data_repaired[i, is_abnormal, :] = avg_vals
            continue

        for j in range(n_features):
            y_good = data_abnormal[i, is_normal, j]
            
            try:
                # 构建插值函数
                # 对于 linear 且非外推场景，np.interp 通常更快
                # 但为了兼容 'cubic' 等方法，使用 scipy.interpolate
                f = interpolate.interp1d(
                    x_good, 
                    y_good, 
                    kind=method, 
                    fill_value="extrapolate",
                    assume_sorted=True  # 性能优化：明确告知 x 是升序的
                )
                
                data_repaired[i, is_abnormal, j] = f(x_bad)
            except Exception:
                # 如果高级插值（如 cubic）失败，自动降级为线性插值
                data_repaired[i, is_abnormal, j] = np.interp(x_bad, x_good, y_good)
                
    return data_repaired