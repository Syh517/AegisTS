import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def kalman_filter_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    基于卡尔曼平滑（Kalman Smoothing）的全局重构修复函数。
    
    逻辑变更：不再依赖 label 指定修复位置，而是将观测数据视为包含测量噪声的信号，
    通过状态空间模型提取其“最可能”的真实状态，从而达到全局修复和去噪的目的。
    """
    data_repaired = np.zeros_like(data_abnormal)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            
            # --- 自动鲁棒性处理 ---
            # 虽然不再强制使用 label，但为了让 EM 算法不被极端尖峰（脏数据）带偏，
            # 我们内部识别一下极值并将其视为隐性缺失（Masked）
            # 这里使用简单的分位数或中值差来识别“极脏”的点
            q25, q75 = np.percentile(series, [25, 75])
            iqr = q75 - q25
            extreme_mask = (series < q25 - 5 * iqr) | (series > q75 + 5 * iqr)
            
            # 使用 MaskedArray 告诉 pykalman 这些点不可信
            masked_series = np.ma.masked_array(series, mask=extreme_mask)
            
            # 基础参数初始化
            initial_mean = np.mean(series[~extreme_mask]) if np.any(~extreme_mask) else 0.0
            initial_cov = np.var(series[~extreme_mask]) if np.any(~extreme_mask) else 1.0

            # 构建模型
            # 采用 Constant Level 模型（随机游走），这是时序平滑最通用的模型
            kf = KalmanFilter(
                initial_state_mean=initial_mean,
                initial_state_covariance=initial_cov,
                transition_matrices=[1.0],
                observation_matrices=[1.0],
                # 初始给一个较大的观测噪声，让模型更相信状态转移，从而更平滑
                observation_covariance=1.0, 
                transition_covariance=0.05
            )

            try:
                # 1. 使用 EM 算法自动调整参数
                # 它会根据序列的波动情况自动平衡 transition_covariance 和 observation_covariance
                kf = kf.em(masked_series, n_iter=5)
                
                # 2. 执行平滑逻辑 (Kalman Smoothing)
                # smooth 会利用整个时间序列（过去和未来）的信息来计算每个点的状态估计
                smoothed_state_means, _ = kf.smooth(masked_series)
                
                # 3. 全局替换：直接使用平滑后的状态作为修复结果
                data_repaired[sample, :, col] = smoothed_state_means.flatten()
                
            except Exception:
                # 备用方案：如果矩阵分解失败，回退到移动平均平滑
                data_repaired[sample, :, col] = pd.Series(series).rolling(
                    window=3, center=True, min_periods=1
                ).mean().values
                
    return data_repaired