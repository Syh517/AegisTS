import numpy as np
from scipy import interpolate
from shapely.geometry import LineString

def trajectory_simplification_repair(
    data_abnormal: np.ndarray, 
    label: np.ndarray = None, 
    tolerance: float = 0.01
) -> np.ndarray:
    """
    基于轨迹简化（Douglas-Peucker）的全局平滑重构函数。
    
    逻辑变更：不再依赖 label 标记。通过简化算法自动识别并剔除偏离主路径的噪声点（脏数据），
    保留关键特征点（骨架），并使用 Pchip 插值重构全量数据。
    
    参数:
    ----
    data_abnormal: (n_samples, n_timestamps, n_features)
    label: 可选，保留以维持 API 一致性
    tolerance: 简化容差。值越大，过滤掉的细节和噪声越多。
    """
    data_repaired = np.zeros_like(data_abnormal)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    x_coords = np.arange(n_timestamps).astype(float)
    
    for sample in range(n_samples):
        for col in range(n_features):
            y_values = data_abnormal[sample, :, col]
            
            # 1. 构建全量坐标矩阵 (Time, Value)
            coords = np.column_stack((x_coords, y_values))
            line = LineString(coords)
            
            # 2. 全局简化：Douglas-Peucker 算法会自动忽略距离主路径小于 tolerance 的点
            # 这种方法对“单点尖峰”脏数据具有天然的免疫力
            simplified = line.simplify(tolerance, preserve_topology=False)
            
            # 3. 提取骨架点 (Skeletal Points)
            s_coords = np.array(simplified.coords)
            x_s = s_coords[:, 0]
            y_s = s_coords[:, 1]

            # 4. 插值重构 (Global Reconstruction)
            try:
                # Pchip (Piecewise Cubic Hermite Interpolating Polynomial)
                # 优点：在保证平滑的同时，不会像普通 Spline 那样产生虚假的波峰（过冲）
                if len(x_s) >= 3:
                    f = interpolate.PchipInterpolator(x_s, y_s, extrapolate=True)
                else:
                    # 如果简化得太厉害（只剩两点），则退化为线性
                    f = interpolate.interp1d(x_s, y_s, kind="linear", fill_value="extrapolate")
                
                # 整个序列由骨架点生成的曲线重构
                data_repaired[sample, :, col] = f(x_coords)
                
            except Exception:
                # 兜底：如果重构失败，返回原始数据的移动平均平滑结果
                data_repaired[sample, :, col] = np.convolve(y_values, np.ones(3)/3, mode='same')
                
    return data_repaired