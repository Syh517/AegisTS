import numpy as np
import warnings
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def impute_svr_univariate(data_missing: np.ndarray, look_back: int = 5) -> np.ndarray:
    """
    使用 NumPy 和 SVR (支持向量回归) 修复缺失值。
    逻辑：初始线性填充 -> 构造滑动窗口数据集 -> SVR 训练 -> 迭代预测 -> 全局均值兜底
    
    :param data_missing: 待修复的 numpy 数组 (N x T x D)
    :param look_back: 用于预测的历史时间步数 (Window Size)
    :return: 修复后的 numpy 数组 (N x T x D)
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    
    # 1. 预处理 inf 值
    data_imputed[np.isinf(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 2. 计算全局特征均值 (D,) 用于整列缺失的极端情况
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本、逐特征处理
    for i in range(n_samples):
        for d in range(n_features):
            series = data_imputed[i, :, d]
            nan_mask = np.isnan(series)
            
            if not np.any(nan_mask):
                continue
                
            # 情况 A: 整列全缺失
            if np.all(nan_mask):
                data_imputed[i, :, d] = global_feature_means[d]
                continue

            # 情况 B: 初始“种子”填充 (Seeding)
            # SVR 训练需要完整序列，np.interp 自动处理中间插值与两端填充(ffill/bfill)
            valid_idx = np.where(~nan_mask)[0]
            seeded_series = series.copy()
            seeded_series[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_idx, series[valid_idx]
            )

            # 如果序列长度不足以构造训练集，直接跳过 SVR
            if n_timesteps <= look_back + 1:
                data_imputed[i, :, d] = seeded_series
                continue

            # 情况 C: 构造 SVR 监督学习数据集
            # 目标：利用 x[t-L:t] 预测 x[t]
            try:
                # 使用 stride_tricks 高效构造滑动窗口，避免 Python 显式循环
                # X_train 形状: (T-look_back, look_back)
                # Y_train 形状: (T-look_back,)
                from numpy.lib.stride_tricks import sliding_window_view
                X_train = sliding_window_view(seeded_series[:-1], window_shape=look_back)
                Y_train = seeded_series[look_back:]

                # 训练 SVR 模型 (带标准化流水线)
                # SVR 对量纲敏感，必须使用 StandardScaler
                model = make_pipeline(
                    StandardScaler(), 
                    SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
                )
                model.fit(X_train, Y_train)

                # 情况 D: 迭代预测缺失值
                # 为了保持逻辑连贯，对于每一个缺失点，使用它之前的最新数据（包含已预测的点）
                series_filled = series.copy()
                # 动态参考数组，用于获取最新的历史值
                dynamic_series = seeded_series.copy()

                for t in range(n_timesteps):
                    if nan_mask[t]:
                        if t < look_back:
                            # 如果缺失点太靠前，无法回溯 look_back 步，保留 seeded 结果
                            series_filled[t] = seeded_series[t]
                        else:
                            # 提取预测特征：t 前面的 look_back 个点
                            x_input = dynamic_series[t - look_back : t].reshape(1, -1)
                            pred_val = model.predict(x_input)[0]
                            series_filled[t] = pred_val
                            # 更新动态数组，使后续预测能利用到当前预测值 (Autoregressive prediction)
                            dynamic_series[t] = pred_val
                
                data_imputed[i, :, d] = series_filled

            except Exception:
                # 若 SVR 失败，则保留线性插值结果
                data_imputed[i, :, d] = seeded_series

    # 4. 最终安全性加固
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed