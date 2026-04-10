import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def autoregressive_repair(
    data_abnormal: np.ndarray, label: np.ndarray, lags=3
) -> np.ndarray:
    """
    优化后的自回归修复函数：
    1. 增加异常占比检查，防止模型学习到噪声
    2. 优化拟合数据，减少插值对AR参数的干扰
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        # 异常掩码
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        # 如果异常点占比超过 70%，AR 模型可能无法捕捉真实趋势，建议跳过或使用简单插值
        if np.mean(is_abnormal) > 0.7:
            # 降级处理：使用线性填充
            for col in range(n_features):
                s = pd.Series(data_abnormal[sample, :, col])
                s[is_abnormal] = np.nan
                data_repaired[sample, :, col] = s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            continue

        for col in range(n_features):
            series_raw = data_abnormal[sample, :, col].copy()
            
            # 策略：为了训练 AR 模型，必须先有一个完整的序列
            # 但我们只在“相对干净”的部分训练，或者对缺失部分进行初步平滑
            series_tmp = pd.Series(series_raw)
            series_tmp[is_abnormal] = np.nan
            # 初步修复用于模型拟合 (使用 linear 避免过度扭曲趋势)
            series_filled = series_tmp.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values

            try:
                # 拟合 AR 模型
                # trend='c' 包含常数项，通常比 'n' 更能适应有漂移的数据
                model = AutoReg(series_filled, lags=lags, trend="c", old_names=False)
                fitted = model.fit()
                
                # 获取全量预测值
                preds = fitted.predict(start=0, end=n_timestamps - 1)
                
                # 仅替换标记为异常的点
                # 注意：AutoReg 预测的前 lags 个值可能为 NaN，需要补齐
                if len(preds) < n_timestamps:
                    # 长度对齐处理
                    full_preds = np.zeros(n_timestamps)
                    full_preds[-len(preds):] = preds
                    full_preds[:-len(preds)] = series_filled[:-len(preds)]
                    preds = full_preds
                
                data_repaired[sample, is_abnormal, col] = preds[is_abnormal]
            except:
                # 若 AR 拟合失败（如序列秩不足），回退到简单线性插值
                data_repaired[sample, is_abnormal, col] = series_filled[is_abnormal]
        
    return data_repaired