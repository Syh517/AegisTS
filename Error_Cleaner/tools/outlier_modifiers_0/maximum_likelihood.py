import numpy as np

def maximum_likelihood_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    优化后的 MLE 修复函数：
    1. 性能：利用 NumPy 向量化操作消除特征级循环，提升效率。
    2. 逻辑：将随机采样改为极大似然估计下的期望值（均值），避免引入随机毛刺。
    3. 鲁棒性：处理全异常或零方差特征。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 将异常点设为 NaN 方便计算
    data_with_nan = data_abnormal.copy().astype(float)
    # 构造三维掩码
    mask_abnormal = (label == 1)
    data_with_nan[mask_abnormal] = np.nan
    
    # 向量化计算每个样本每个特征的 MLE 参数 (mu)
    # 计算忽略 NaN 的均值，axis=1 代表时间轴
    with np.errstate(all='ignore'):
        mus = np.nanmean(data_with_nan, axis=1) # 形状: (n_samples, n_features)
        
        # 填充均值中的无效值（如果某个特征全是 NaN，均值为 NaN）
        # 使用全局均值或 0 作为保底
        mus = np.nan_to_num(mus, nan=0.0)
    
    # 修复逻辑
    for i in range(n_samples):
        # 提取当前样本的异常掩码
        sample_mask = mask_abnormal[i]
        if not np.any(sample_mask):
            continue
            
        # 修复方式选择：
        # 真正的 MLE 修复在没有额外约束时，异常点最可能的取值就是该分布的均值
        # 这里直接将该样本该特征的均值赋予异常点，保证统计中心不偏移
        # 形状：(异常点数量, n_features)
        fill_values = mus[i] 
        
        # 广播赋值：将均值填入该样本的所有异常点位置
        # data_repaired[i, sample_mask] 形状为 (n_bad_points, n_features)
        data_repaired[i, sample_mask, :] = fill_values
        
    return data_repaired