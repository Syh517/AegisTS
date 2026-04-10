import numpy as np

def moving_average_repair(
    data: np.ndarray, 
    label: np.ndarray,
    window_size: int = 5, 
    mode: str = 'same'
) -> np.ndarray:
    """
    基于移动平均的多变量时序平滑修复函数
    
    参数:
    ----
    data: np.ndarray
        输入数据，形状为 (n_samples, n_timestamps, n_features)
    window_size: int
        滑动窗口的大小，值越大平滑力度越强，但会丢失更多细节
    mode: str
        卷积模式，'same' 保证输出长度不变
        
    返回:
    ----
    data_smoothed: np.ndarray
        平滑处理后的数组 (n_samples, n_timestamps, n_features)
    """
    n_samples, n_timestamps, n_features = data.shape
    data_smoothed = np.zeros_like(data)
    
    # 定义移动平均核
    kernel = np.ones(window_size) / window_size
    
    for i in range(n_samples):
        for j in range(n_features):
            # 获取单条序列
            series = data[i, :, j]
            
            # 使用 np.convolve 进行快速卷积运算
            # 'same' 模式会自动处理中心对齐，但在边缘可能存在偏差
            smoothed_series = np.convolve(series, kernel, mode='same')
            
            # --- 边界处理优化 ---
            # 卷积的 'same' 模式在两端会因为补零而导致数值下降
            # 我们用边缘值填充来修复两端的平滑效果
            pad_size = window_size // 2
            if pad_size > 0:
                # 起始端补齐
                smoothed_series[:pad_size] = np.mean(series[:window_size])
                # 末尾端补齐
                smoothed_series[-pad_size:] = np.mean(series[-window_size:])
            
            data_smoothed[i, :, j] = smoothed_series
            
    return data_smoothed