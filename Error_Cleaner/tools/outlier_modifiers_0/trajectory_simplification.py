import numpy as np
from scipy import interpolate
from shapely.geometry import LineString

def trajectory_simplification_repair(
    data_abnormal: np.ndarray, label: np.ndarray, tolerance=0.01
) -> np.ndarray:
    """
    优化后的轨迹简化修复函数
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    x_coords = np.arange(n_timestamps).astype(float)
    
    for sample in range(n_samples):
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        is_normal = ~is_abnormal
        # 提取当前样本的正常点索引
        x_normal = x_coords[is_normal]
        
        # 基础校验
        if len(x_normal) < 2:
            # 若正常点太少，尝试用均值填充异常位
            fill_val = np.nanmean(data_abnormal[sample], axis=0)
            data_repaired[sample, is_abnormal, :] = fill_val
            continue

        for col in range(n_features):
            y_normal = data_abnormal[sample, is_normal, col]
            
            # 1. 轨迹简化：仅在数据点非常密集或噪声极大时有意义
            # 将 numpy 数组转为坐标矩阵，利用 Shapely 向量化接口提升速度
            coords = np.column_stack((x_normal, y_normal))
            line = LineString(coords)
            
            # preserve_topology=False 加速计算
            simplified = line.simplify(tolerance, preserve_topology=False)
            
            # 2. 提取骨架点
            # 直接从 simplified.coords 获取，比访问 .xy 属性更快
            s_coords = np.array(simplified.coords)
            x_s = s_coords[:, 0]
            y_s = s_coords[:, 1]

            # 3. 插值重建
            # 使用 Pchip 替代 linear：Pchip 能够保持数据的单调性，防止插值产生的“过冲”
            # 且比普通的样条插值更符合物理轨迹规律
            try:
                # 如果骨架点足够，使用 Pchip
                if len(x_s) >= 3:
                    f = interpolate.PchipInterpolator(x_s, y_s, extrapolate=True)
                else:
                    f = interpolate.interp1d(x_s, y_s, kind="linear", fill_value="extrapolate")
                
                # 4. 修复异常点
                x_bad = x_coords[is_abnormal]
                data_repaired[sample, is_abnormal, col] = f(x_bad)
                
            except Exception:
                # 兜底：线性插值
                data_repaired[sample, is_abnormal, col] = np.interp(
                    x_coords[is_abnormal], x_normal, y_normal
                )
                
    return data_repaired