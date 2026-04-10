import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import warnings

def state_space_repair(data_abnormal: np.ndarray, label: np.ndarray = None) -> np.ndarray:
    """
    基于状态空间模型（Local Level Model）的全局平滑修复函数。
    
    逻辑变更：不再依据 label 局部替换，而是将整个多变量序列视为受污染的观测，
    通过 SSM 提取最可能的平滑隐状态轨迹，实现全局重构。
    """
    # 忽略计算中的微小警告
    warnings.filterwarnings("ignore")
    
    data_repaired = np.zeros_like(data_abnormal)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            
            # --- 鲁棒性预处理：自动化隐式掩码 ---
            # 为了防止 EM 算法被脏数据点彻底带偏，我们依然需要识别出极度离群的点
            # 即使没有外部 label，模型也应该具备识别“不可信点”的能力
            median = np.median(series)
            std = np.std(series)
            # 识别偏差超过 4 倍标准差的点，视为临时缺失（Masked）
            auto_mask = np.abs(series - median) > (4 * std)
            
            # 关键优化：使用 Numpy 掩码数组处理
            masked_series = np.ma.masked_array(series, mask=auto_mask)

            # 初始化均值：使用平稳部分的首个有效值
            initial_mean = series[~auto_mask][0] if np.any(~auto_mask) else 0.0

            # 构建状态空间模型
            # transition_matrices=[1] 代表随机游走模型，是最稳健的平滑基准
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_mean,
                initial_state_covariance=1.0,
                observation_covariance=1.0,  # 初始观测方差
                transition_covariance=0.01   # 初始过程方差
            )

            try:
                # 1. 参数调优：通过 EM 算法从数据中学习真正的噪声水平
                # 脏数据越多，EM 学到的 observation_covariance 越大，平滑力度就越强
                kf = kf.em(masked_series, n_iter=5)
                
                # 2. 全局平滑 (State Space Smoothing)
                # 利用整条序列的上下文信息提取最优隐状态
                state_means, _ = kf.smooth(masked_series)
                
                # 3. 全局重构：将原始序列直接替换为平滑后的状态均值
                data_repaired[sample, :, col] = state_means.flatten()
                
            except Exception:
                # 极端失败降级：移动中值滤波平滑
                data_repaired[sample, :, col] = pd.Series(series).rolling(
                    window=5, center=True, min_periods=1
                ).median().values
        
    return data_repaired