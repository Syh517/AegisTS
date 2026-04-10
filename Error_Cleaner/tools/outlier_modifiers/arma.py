import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

def arma_repair(
    data_abnormal: np.ndarray, 
    label: np.ndarray, 
    p: int = 2, 
    q: int = 1
) -> np.ndarray:
    """
    基于 ARMA (p, q) 模型的全局平滑修复函数。
    
    逻辑变更：不再仅修复 label 标记处，而是通过 ARMA 模型的拟合值对整条序列进行重构，
    利用 AR 的趋势捕获能力和 MA 的噪声平滑能力来“修复”脏数据。
    """
    # 忽略模型收敛警告
    warnings.filterwarnings("ignore")
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    data_repaired = np.zeros_like(data_abnormal)
    
    for sample in range(n_samples):
        for col in range(n_features):
            series_raw = data_abnormal[sample, :, col].copy()
            
            # 1. 鲁棒性预处理
            # 即使不使用 label，我们先用中值平滑掉极端离群点，
            # 这样 ARIMA 在 fit 时捕获的参数 phi 和 theta 才是真正反映信号特征的
            series_pd = pd.Series(series_raw)
            series_interp = series_pd.rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill').values

            try:
                # 2. 拟合 ARMA(p, q)
                # order=(p, 0, q) 等效于 ARMA(p, q)
                # 使用 simple_differencing=False 和状态空间求解，提高复杂序列下的收敛性
                model = ARIMA(series_interp, order=(p, 0, q), 
                              enforce_stationarity=False, 
                              enforce_invertibility=False)
                
                fitted = model.fit()
                
                # 3. 全量预测（重构）
                # predict(0, n-1) 会生成一条基于 ARMA 参数的“理想”平滑曲线
                preds = fitted.predict(start=0, end=n_timestamps - 1)
                
                # 4. 赋值修复后的序列
                data_repaired[sample, :, col] = preds
                
            except Exception:
                # 极端情况下若 ARMA 拟合崩溃，返回初步平滑的中值序列
                data_repaired[sample, :, col] = series_interp
                
    return data_repaired