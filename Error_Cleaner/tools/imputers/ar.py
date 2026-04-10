import numpy as np
import warnings
from statsmodels.tsa.ar_model import AutoReg

def impute_ar(data_missing: np.ndarray, lags: int = 1) -> np.ndarray:
    """
    优化后的 AR 插补法：增加了稳定性检查、数值截断和多级回退机制。
    
    :param data_missing: 待修复的 numpy 数组 (N x T x D)
    :param lags: AR 模型的滞后阶数
    :return: 修复后的 numpy 数组 (N x T x D)，保证无 NaN/Inf
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    # 1. 初始化与预处理
    data_imputed = np.copy(data_missing)
    # 将所有 inf 转为 nan，统一处理
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 2. 计算全局特征均值 (D,) 作为最终兜底方案
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        # 如果某一特征全为 NaN，则填充为 0.0
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本、逐特征处理
    for i in range(n_samples):
        for d in range(n_features):
            series = data_imputed[i, :, d]
            nan_mask = np.isnan(series)
            
            if not np.any(nan_mask):
                continue
                
            # 情况 A: 整列全缺失 -> 直接用特征均值
            if np.all(nan_mask):
                data_imputed[i, :, d] = global_feature_means[d]
                continue
            
            # 获取有效数据的范围，用于后续数值截断 (Clipping)
            valid_vals = series[~nan_mask]
            s_min, s_max = np.min(valid_vals), np.max(valid_vals)
            
            # 情况 B: 部分缺失
            series_filled = series.copy()
            filled_flag = False # 标记是否成功通过 AR 填充
            
            # --- 尝试 AR 模型插补 ---
            if len(valid_vals) > lags + 1:
                try:
                    # 准备临时序列用于训练（AR不能处理含有NaN的训练集）
                    # 先用线性插值填充临时序列
                    valid_coords = np.where(~nan_mask)[0]
                    temp_series = np.interp(
                        np.arange(n_timesteps), valid_coords, valid_vals
                    )
                    
                    # 拟合 AR 模型
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = AutoReg(temp_series, lags=lags, trend='c', old_names=False).fit()
                        preds = model.predict(start=0, end=n_timesteps - 1)
                    
                    # 稳定性检查：预测结果必须是有限值
                    if np.all(np.isfinite(preds)):
                        # 数值截断：防止 AR 模型预测出脱离实际范围的巨量值
                        preds = np.clip(preds, s_min, s_max)
                        series_filled[nan_mask] = preds[nan_mask]
                        filled_flag = True
                except:
                    # 模型拟合失败（如矩阵奇异），filled_flag 保持 False
                    pass

            # --- 情况 C: AR 失败或点数不足 -> 回退到线性插值 ---
            if not filled_flag:
                valid_coords = np.where(~nan_mask)[0]
                # np.interp 会自动处理首尾：缺失的首部补第一个有效值，尾部补最后一个
                interp_series = np.interp(
                    np.arange(n_timesteps), valid_coords, valid_vals
                )
                series_filled[nan_mask] = interp_series[nan_mask]

            # --- 情况 D: 最终安全性检查 ---
            # 处理可能出现的极端情况（如 interp 返回了非有限值）
            final_nan_mask = ~np.isfinite(series_filled)
            if np.any(final_nan_mask):
                series_filled[final_nan_mask] = global_feature_means[d]
            
            data_imputed[i, :, d] = series_filled

    # 4. 彻底消除残余的极小概率异常值
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed