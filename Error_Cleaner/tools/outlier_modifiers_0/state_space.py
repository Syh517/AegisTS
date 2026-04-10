import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import warnings

def state_space_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    优化后的状态空间修复函数：
    1. 使用 MaskedArray 替代预插值，保留原始不确定性。
    2. 优化 EM 初始化参数逻辑。
    3. 提高在大规模数据下的运行鲁棒性。
    """
    # 忽略计算中的微小警告
    warnings.filterwarnings("ignore")
    
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        # 异常标记
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            
            # 关键优化：使用 Numpy 掩码数组。
            # pykalman 的 EM 算法会自动处理掩码部分，将其视为缺失观测，
            # 从而通过状态方程进行纯预测推断。
            masked_series = np.ma.masked_array(series, mask=is_abnormal)

            # 初始化均值：取第一个有效观测值
            valid_indices = np.where(~is_abnormal)[0]
            if len(valid_indices) == 0:
                # 若无有效值，使用全序列均值或0
                initial_mean = np.nanmean(series) if not np.all(np.isnan(series)) else 0.0
            else:
                initial_mean = series[valid_indices[0]]

            # 构建状态空间模型 (Local Level Model)
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_mean,
                initial_state_covariance=1.0,
                # 预设合理的方差比例，加速 EM 收敛
                observation_covariance=1.0,
                transition_covariance=0.05 
            )

            try:
                # 优化：EM 迭代次数设为 5 次，平衡精度与速度
                # 它会根据 masked_series 中的有效点自动调整 Q 和 R
                kf = kf.em(masked_series, n_iter=5)
                
                # 平滑处理：利用双向信息修复异常点
                state_means, _ = kf.smooth(masked_series)
                
                # 提取修复值
                repair_values = state_means.flatten()
                data_repaired[sample, is_abnormal, col] = repair_values[is_abnormal]
                
            except Exception:
                # 极端失败情况下的降级：线性插值
                s_pd = pd.Series(series)
                s_pd[is_abnormal] = np.nan
                data_repaired[sample, is_abnormal, col] = s_pd.interpolate().fillna(method="bfill").fillna(method="ffill").values[is_abnormal]
        
    return data_repaired