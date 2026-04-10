import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

def exponential_smoothing_repair(
    data_abnormal: np.ndarray, 
    label: np.ndarray = None, 
    alpha: float = 0.5
) -> np.ndarray:
    """
    基于指数平滑（Holt-Winters）的全局重构修复函数。
    
    逻辑变更：抛弃基于 label 的局部替换。利用指数平滑对整条序列进行重新计算，
    通过调整 alpha（平滑系数）来过滤脏数据中的高频噪声。
    
    参数:
    ----
    data_abnormal: (n_samples, n_timestamps, n_features)
    label: 可选，保留以维持 API 兼容性。
    alpha: 平滑水平 (0 < alpha <= 1)。值越小，平滑力度越大，对历史数据的依赖越重。
    """
    # 忽略某些序列因波动极小导致的收敛警告
    warnings.filterwarnings("ignore")
    
    data_repaired = np.zeros_like(data_abnormal)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            
            # --- 鲁棒性增强：中值预处理 ---
            # 为了防止指数平滑在序列开始处的极端点（脏数据）导致整体偏移，
            # 我们先进行轻微的中值滤波以剔除尖峰脉冲。
            series_pd = pd.Series(series)
            series_pre = series_pd.rolling(window=3, center=True, min_periods=1).median().values

            try:
                # 使用简单的指数平滑 (Simple Exponential Smoothing)
                # 如果数据有明显的趋势或季节性，可以将 trend 改为 'add'
                model = ExponentialSmoothing(series_pre, trend=None, seasonal=None)
                
                # 拟合模型
                # smoothing_level 即为 alpha：决定了新观测值与旧状态的权重分配
                fitted = model.fit(smoothing_level=alpha, optimized=False)
                
                # 获取完整的拟合序列 (Fitted Values)
                # 这条序列已经是经过加权平滑后的结果
                data_repaired[sample, :, col] = fitted.fittedvalues
                
            except Exception:
                # 如果拟合失败（如数据量过少），回退到原始的中值预处理结果
                data_repaired[sample, :, col] = series_pre
                
    return data_repaired