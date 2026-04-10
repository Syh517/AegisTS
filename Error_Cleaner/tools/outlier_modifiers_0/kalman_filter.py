import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def kalman_filter_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    优化后的卡尔曼滤波修复函数：
    1. 引入多变量协同可能性的预留（当前保持逐特征以维持API一致性）。
    2. 优化 EM 学习逻辑：仅在必要时学习，减少迭代次数。
    3. 增强稳定性：通过掩码处理 MaskedArray，这是 pykalman 处理 NaN 的推荐方式。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        # 异常掩码
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            
            # 使用 MaskedArray 是 pykalman 处理缺失值（NaN/异常点）的标准做法
            # 这比直接传入 NaN 并在内部转换更高效、更稳定
            masked_series = np.ma.masked_array(series, mask=is_abnormal)
            
            # 基础参数估计
            valid_data = series[~is_abnormal]
            if len(valid_data) < 2:
                # 极端情况：数据几乎全为异常，回退至全局均值或邻域填充
                fill_val = np.nanmean(series) if not np.all(np.isnan(series)) else 0.0
                data_repaired[sample, is_abnormal, col] = fill_val
                continue

            initial_mean = valid_data[0]
            initial_cov = np.var(valid_data) + 1e-6

            # 构建平滑模型
            # transition_matrices=1.0 适用于随机游走模型
            # 如果数据有明显的线性趋势，可以考虑设置更复杂的矩阵
            kf = KalmanFilter(
                initial_state_mean=initial_mean,
                initial_state_covariance=initial_cov,
                transition_matrices=[1.0],
                observation_matrices=[1.0],
                observation_covariance=1.0, 
                transition_covariance=0.01  # 略微增大过程噪声，提高对变化的追踪灵敏度
            )

            try:
                # 优化点：减少 EM 迭代次数（从10减至5），通常 5 次即可获得较好的超参数
                # 对于实时性要求高的场景，甚至可以跳过 EM 直接使用启发式参数
                kf = kf.em(masked_series, n_iter=5)
                
                # 使用 smooth 而非 filter，因为平滑利用了“未来”的信息，修复效果更好
                smoothed_state_means, _ = kf.smooth(masked_series)
                
                # 修复数据
                repair_values = smoothed_state_means.flatten()
                data_repaired[sample, is_abnormal, col] = repair_values[is_abnormal]
                
            except Exception:
                # 备用方案：线性插值
                s_pd = pd.Series(series)
                s_pd[is_abnormal] = np.nan
                data_repaired[sample, is_abnormal, col] = s_pd.interpolate(
                    method="linear", limit_direction="both"
                ).values[is_abnormal]
                
    return data_repaired