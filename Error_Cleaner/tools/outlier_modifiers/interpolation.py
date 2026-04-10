import numpy as np
from scipy import interpolate

def interpolation_repair(
    data_abnormal: np.ndarray, 
    label: np.ndarray = None, 
    method: str = "linear",
    sampling_step: int = 3
) -> np.ndarray:
    """
    基于重构插值的多变量时序修复函数。
    
    逻辑变更：不再依赖 label 标记。通过在序列中提取子样本（采样点）作为“可信基准”，
    利用插值函数重构出全量时间点，从而达到过滤噪声和脏数据的平滑效果。
    
    参数:
    ----
    data_abnormal: (n_samples, n_timestamps, n_features)
    label: 可选，保留以维持 API 兼容性（内部可不使用或作为先验参考）
    method: 'linear', 'quadratic', 'cubic', 'slinear'
    sampling_step: 采样步长。值越大，平滑力度越强（忽略的细节越多）。
    """
    n_samples, n_timestamps, n_features = data_abnormal.shape
    data_repaired = np.zeros_like(data_abnormal)
    
    # 全量时间索引
    x_full = np.arange(n_timestamps)
    # 采样索引：作为插值的支撑点（Knot points）
    x_sampled = x_full[::sampling_step]
    
    # 确保最后一个点被包含，防止尾部外推失真
    if x_sampled[-1] != x_full[-1]:
        x_sampled = np.append(x_sampled, x_full[-1])

    for i in range(n_samples):
        for j in range(n_features):
            series = data_abnormal[i, :, j]
            
            # 提取支撑点的数值
            y_sampled = series[x_sampled]
            
            try:
                # 构造插值函数
                # 这里本质上是将 series 降采样后再升采样回原长度
                f = interpolate.interp1d(
                    x_sampled, 
                    y_sampled, 
                    kind=method, 
                    fill_value="extrapolate",
                    assume_sorted=True
                )
                
                # 重构整条序列
                # 那些偏离插值曲线的“脏数据点”会被 f(x_full) 修正
                data_repaired[i, :, j] = f(x_full)
                
            except Exception:
                # 降级处理：使用 NumPy 原生线性插值
                data_repaired[i, :, j] = np.interp(x_full, x_sampled, y_sampled)
                
    return data_repaired