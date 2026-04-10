import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def svr_repair(data_abnormal: np.ndarray, label: np.ndarray = None) -> np.ndarray:
    """
    基于 SVR (支持向量回归) 的全局趋势重构修复函数。
    
    逻辑变更：抛弃基于 label 的局部填补。利用 SVR 的软间隔特性对全量数据进行拟合。
    SVR 会自动识别哪些是“支撑向量”，并忽略偏离主趋势的“脏数据点”（视为噪声），
    从而重构出一条极其稳健的全局平滑曲线。
    """
    data_repaired = np.zeros_like(data_abnormal)
    n_samples, n_timestamps, n_features = data_abnormal.shape

    # 1. 时间索引预处理：标准化对 SVR 性能至关重要
    x = np.arange(n_timestamps).reshape(-1, 1)
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)

    for sample in range(n_samples):
        for col in range(n_features):
            y = data_abnormal[sample, :, col].reshape(-1, 1)
            
            # --- 鲁棒性增强：中值预清洗 ---
            # 即使 SVR 抗噪，但如果有量级巨大的脏数据（如 100 倍跳变），仍会拉偏支持向量。
            # 先进行极轻微的中值平滑作为训练输入。
            y_series = pd.Series(y.ravel())
            y_pre = y_series.rolling(window=3, center=True, min_periods=1).median().values.reshape(-1, 1)

            try:
                # 2. 目标值标准化：使 C 和 epsilon 参数具有通用性
                y_scaler = StandardScaler()
                y_scaled = y_scaler.fit_transform(y_pre).ravel()

                # 3. 配置 SVR 全局回归：
                # C=1.0: 惩罚系数。值越小，平滑度越高。
                # epsilon=0.1: 容忍度。该范围内的噪声会被忽略。
                # kernel="rbf": 能够处理非线性的复杂趋势。
                svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma='scale', cache_size=1000)
                
                # 直接对全量序列进行拟合
                svr.fit(x_scaled, y_scaled)
                
                # 4. 全局重构预测
                y_pred_scaled = svr.predict(x_scaled)
                
                # 5. 反标准化回原始量程
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                data_repaired[sample, :, col] = y_pred

            except Exception:
                # 极端失败降级：使用移动平均
                data_repaired[sample, :, col] = y_series.rolling(window=5, center=True, min_periods=1).mean().values

    return data_repaired