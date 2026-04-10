import numpy as np
from scipy.ndimage import median_filter

def ols_repair(
    data_abnormal: np.ndarray, 
    p: int = 3, 
    delta: float = 1e-5, 
    max_iter: int = 100
) -> np.ndarray:
    """
    改造后的统计 OLS 修复函数 (无 Label 版):
    1. 仿照 Java OLS 逻辑: 针对残差 z 建模, 并在迭代中动态更新 xMatrix。
    2. 统计识别: 利用预测残差与实际残差的 L1 距离自动定位离群点。
    3. 最小修改原则: 每一轮迭代修复偏差最大的点，模拟 mainOLS 的 repairAMin 逻辑。
    """
    # 1. 强制将 p 转换为整数，防止外部传入 float 或 array
    p = int(np.asarray(p).item()) 
    
    n_samples, n_timesteps, n_features = data_abnormal.shape
    data_repaired = data_abnormal.copy().astype(float)

    for s in range(n_samples):
        for c in range(n_features):
            series = data_repaired[s, :, c]
            
            # --- 去趋势 ---
            baseline = median_filter(series, size=min(n_timesteps, 11))
            z = series - baseline
            
            row_num = n_timesteps - p
            if row_num <= 0: continue

            # --- 2. 矩阵化重构 (替代原报错的列表推导式) ---
            # 使用 NumPy 的 stride_tricks 构造滑动窗口矩阵，更安全且极快
            from numpy.lib.stride_tricks import as_strided
            
            # 构造 x_matrix: 每一行是 [z[t-1], z[t-2], ..., z[t-p]]
            # 构造 y_matrix: 对应 z[t]
            x_matrix = as_strided(
                z, 
                shape=(row_num, p), 
                strides=(z.strides[0], z.strides[0])
            )[:, ::-1].copy() # 这里的 copy 确保内存连续
            
            y_matrix = z[p:].reshape(row_num, 1).copy()

            while iteration_num < max_iter:
                iteration_num += 1
                
                # --- 2. 学习参数 phi (OLS 最小二乘) ---
                # 使用伪逆解决奇异矩阵问题，对应 Java 中的 Matrix.solve 或 pinv
                try:
                    phi = np.linalg.pinv(x_matrix.T @ x_matrix) @ x_matrix.T @ y_matrix
                except np.linalg.LinAlgError:
                    break
                
                # --- 3. 计算预测残差 y_hat ---
                y_hat = x_matrix @ phi
                
                # --- 4. 统计识别 (替代原 repairAMin) ---
                # 计算创新项 (Innovation): 观测残差与模型预测残差的绝对差值
                innovations = np.abs(y_matrix - y_hat).flatten()
                
                # 使用稳健统计阈值 (MAD) 判定
                mad = np.median(np.abs(innovations - np.median(innovations)))
                threshold = max(mad * 3.0, 1e-6)
                
                # 寻找偏离最大的点进行修复 (IMR 核心逻辑)
                target_index = np.argmax(innovations)
                
                # 如果最大偏差点都在统计容忍范围内，说明收敛
                if innovations[target_index] < threshold:
                    break
                
                # --- 5. 更新逻辑 (同步 Java OLS 的局部更新) ---
                old_val = y_matrix[target_index, 0]
                new_val = y_hat[target_index, 0]
                
                if np.abs(old_val - new_val) < delta:
                    break
                
                # 更新 y_matrix
                y_matrix[target_index, 0] = new_val
                
                # 更新 x_matrix 中受影响的后续滑动窗口
                # 对应 Java 版: for (int j = 0; j < p; ++j) { i = index + 1 + j; x[i][j] = val; }
                for j in range(p):
                    i = target_index + 1 + j
                    if i < row_num:
                        x_matrix[i, j] = new_val
                
            # --- 6. 还原修复后的数据 ---
            # y_repaired = baseline + z_repaired
            data_repaired[s, p:, c] = baseline[p:] + y_matrix.flatten()
            
    return data_repaired