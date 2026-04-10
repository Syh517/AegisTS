import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def autoregressive_repair(
    data: np.ndarray, 
    label: np.ndarray,
    lags: int = 3, 
    pre_smooth: bool = True
) -> np.ndarray:
    """
    基于自回归重构的多变量时序平滑修复函数
    
    参数:
    ----
    data: (n_samples, n_timestamps, n_features)
    lags: AR模型的阶数，即参考过去多少个时间点
    pre_smooth: 是否先进行轻量级中值滤波以防止AR模型被极端异常点干扰
    """
    n_samples, n_timestamps, n_features = data.shape
    data_repaired = np.zeros_like(data)
    
    for i in range(n_samples):
        for j in range(n_features):
            series = data[i, :, j].copy()
            
            # 1. 鲁棒性预处理：如果数据非常脏，AR模型会拟合出错误的参数
            # 使用中值滤波快速剔除极端尖峰，但不改变原始数组
            if pre_smooth:
                # 简单快速的移动中值，防止离群值直接拉偏AR系数
                fit_series = pd.Series(series).rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill').values
            else:
                fit_series = series

            try:
                # 2. 拟合 AR 模型
                # trend='c' 包含常数项，适应有偏移的数据
                model = AutoReg(fit_series, lags=lags, trend="c", old_names=False)
                model_fit = model.fit()
                
                # 3. 获取全量拟合值（重构序列）
                # 注意：AutoReg 默认从第 lags 个点开始预测
                preds = model_fit.predict(start=0, end=n_timestamps - 1)
                
                # 4. 长度对齐与缺失值处理
                # AutoReg 的前几个预测值可能因滞后项不足而效果较差，这里做衔接处理
                if len(preds) < n_timestamps:
                    full_preds = np.zeros(n_timestamps)
                    # 填充预测部分
                    full_preds[-len(preds):] = preds
                    # 起始部分使用原始数据（或预处理后的数据）补齐
                    full_preds[:-len(preds)] = fit_series[:-len(preds)]
                    data_repaired[i, :, j] = full_preds
                else:
                    data_repaired[i, :, j] = preds
                    
            except Exception:
                # 如果 AR 拟合失败（如数据全为常数），则保持原样或返回预处理后的结果
                data_repaired[i, :, j] = fit_series
                
    return data_repaired