import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

def arma_repair(
    data_abnormal: np.ndarray, label: np.ndarray, p=2, q=1
) -> np.ndarray:
    """
    使用ARMA模型修复异常值。
    
    优化点：
    1. 增加异常占比预警与自动降级。
    2. 优化模型参数设置（使用状态空间模型加速）。
    3. 修复原代码中 fittedvalues 可能存在的偏移问题。
    """
    # 忽略模型收敛警告，避免控制台被淹没
    warnings.filterwarnings("ignore")
    
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        # 异常标记 mask (1为异常)
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        # 性能与逻辑优化：如果异常比例过高，ARMA 拟合无意义
        if np.mean(is_abnormal) > 0.6:
            for col in range(n_features):
                s = pd.Series(data_abnormal[sample, :, col])
                s[is_abnormal] = np.nan
                data_repaired[sample, :, col] = s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            continue

        for col in range(n_features):
            series_raw = data_abnormal[sample, :, col].copy()
            
            # 预处理：先用简单插值临时填充，为 ARMA 提供连续输入
            series_pd = pd.Series(series_raw)
            series_pd[is_abnormal] = np.nan
            series_interp = series_pd.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

            try:
                # 优化点：使用 ARIMA(p, 0, q) 等效于 ARMA(p, q)
                # enforce_stationarity=False 增加模型稳定性
                # enforce_invertibility=False 防止 MA 部分计算崩溃
                model = ARIMA(series_interp, order=(p, 0, q), 
                              enforce_stationarity=False, 
                              enforce_invertibility=False)
                
                # 使用 SARIMAX 框架下的优化器，速度通常比旧版 ARMA 快
                fitted = model.fit()
                
                # 使用 predict 而非 fittedvalues 以确保获取完整的样本内重建
                preds = fitted.predict(start=0, end=n_timestamps - 1)
                
                # 仅修复异常部分
                data_repaired[sample, is_abnormal, col] = preds[is_abnormal]
                
            except Exception:
                # 极端情况下（如数据全为常数），ARMA 无法拟合，回退到线性插值结果
                data_repaired[sample, is_abnormal, col] = series_interp[is_abnormal]
                
    return data_repaired