import numpy as np
import pandas as pd

def smurf_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, window=5
) -> np.ndarray:
    """
    基于局部极大似然估计的自适应 SMURF 修复：
    1. 自动检测：利用滑动窗口统计量 (Rolling Mean/Std) 构建概率置信区间。
    2. 局部修复：识别离群点并利用窗口内高概率分布的期望值（局部均值）进行替代。
    3. 移除依赖：完全不使用外部 label，通过局部统计特性自识别“脏数据”。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for s in range(n_samples):
        for c in range(n_features):
            series = pd.Series(data_abnormal[s, :, c])
            
            # --- 1. 计算局部统计量 (Rolling Statistics) ---
            # 使用较宽的窗口来获得稳健的均值和标准差估计
            rolling = series.rolling(window=2*window+1, center=True, min_periods=1)
            local_mean = rolling.mean()
            # 使用稳健的标准差估计（防止离群点过度拉大标准差）
            # 这里可以用差分的中位数来辅助估计 sigma
            local_std = rolling.std().fillna(method='bfill').fillna(method='ffill')
            
            # --- 2. 局部 MLE 检测逻辑 ---
            # 计算残差（实际值与局部期望的偏差）
            residuals = np.abs(series - local_mean)
            
            # 动态阈值：3倍标准差（99.7% 置信区间）
            # 为防止 sigma 为 0 导致失效，加入一个全局的基础 sigma
            global_sigma = np.std(series) * 0.1
            is_dirty = residuals > (3 * local_std + global_sigma)
            
            # --- 3. 极大似然修复 ---
            if is_dirty.any():
                # 识别到的脏数据，其似然度最高的值即为局部均值
                repaired_series = series.copy()
                repaired_series[is_dirty] = local_mean[is_dirty]
                
                # 处理可能产生的连续 NaN（如果整个窗口都脏，导致 local_mean 失效）
                if repaired_series.isnull().any():
                    repaired_series = repaired_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                
                data_repaired[s, :, c] = repaired_series.values
                
    return data_repaired