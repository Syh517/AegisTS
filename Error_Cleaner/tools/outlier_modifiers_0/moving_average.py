import numpy as np
from scipy import interpolate

def moving_average_repair(
    data: np.ndarray, 
    label: np.ndarray = None, 
    sigma: float = 3.0, 
    method: str = 'linear'
) -> np.ndarray:
    """
    优化后的多变量时序清洗函数
    
    参数:
        data: (n_samples, n_timestamps, n_features)
        label: 可选, 预知的异常标记 (n_samples, n_timestamps)
        sigma: 自动检测阈值 (Z-Score), 默认为 3
        method: 修复方法, 'linear', 'nearest', 'cubic'
        
    返回:
        data_repaired: 修复后的数组
    """
    n_samples, n_timestamps, n_features = data.shape
    data_repaired = data.copy()
    
    # 1. 自动异常检测 (如果未提供label)
    if label is None:
        # 向量化计算 Z-Score: (x - mean) / std
        # 对每个样本的每个特征独立计算
        means = np.nanmean(data, axis=1, keepdims=True)
        stds = np.nanstd(data, axis=1, keepdims=True)
        # 避免除以0
        stds[stds == 0] = 1e-6
        z_scores = np.abs((data - means) / stds)
        # 只要有一个特征异常，该时间点标记为异常 (或根据需求修改为逐特征检测)
        combined_label = np.any(z_scores > sigma, axis=2)
    else:
        combined_label = label.astype(bool)

    # 2. 向量化修复逻辑
    # 时间轴索引
    x = np.arange(n_timestamps)
    
    for i in range(n_samples):
        mask = combined_label[i]
        if not np.any(mask):
            continue
            
        # 正常点的索引
        x_good = x[~mask]
        # 异常点的索引
        x_bad = x[mask]
        
        # 如果整条序列都坏了或正常点太少，跳过
        if len(x_good) < 2:
            continue
            
        for j in range(n_features):
            y_good = data[i, ~mask, j]
            
            # 使用插值代替移动平均，能够处理连续异常点
            f = interpolate.interp1d(x_good, y_good, kind=method, fill_value="extrapolate")
            data_repaired[i, mask, j] = f(x_bad)
            
    return data_repaired