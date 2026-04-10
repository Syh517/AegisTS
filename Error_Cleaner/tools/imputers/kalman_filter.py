import numpy as np
import warnings
from pykalman import KalmanFilter

def impute_kalman_filter(data: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    优化后的 Kalman Filter 插补法：增加了稳定性检查、数值截断和健壮性逻辑。
    
    :param data: 待修复的 numpy 数组 (N x T x D)
    :param verbose: 是否打印进度
    :return: 修复后的 numpy 数组 (N x T x D)
    """
    assert len(data.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    # 1. 初始化与预处理：将所有非有限值 (inf, -inf) 统一转为 NaN
    data_imputed = data.copy()
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 2. 计算全局特征均值 (D,) 作为最终兜底
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        # 彻底补全：如果某特征全是 NaN，设为 0.0
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本处理
    for i in range(n_samples):
        for d in range(n_features):
            series = data_imputed[i, :, d]
            nan_mask = np.isnan(series)
            
            if not np.any(nan_mask):
                continue
                
            # 情况 A: 序列完全缺失 -> 使用全局特征均值
            if np.all(nan_mask):
                data_imputed[i, :, d] = global_feature_means[d]
                continue

            # 获取当前序列的有效范围，用于数值截断 (Clip)
            valid_vals = series[~nan_mask]
            s_min, s_max = np.min(valid_vals), np.max(valid_vals)
            
            series_filled = series.copy()
            success_flag = False

            # 情况 B: 尝试 Kalman 平滑
            try:
                # 初始线性填充（Kalman EM 算法对初始值敏感）
                valid_idx = np.where(~nan_mask)[0]
                temp_interp = np.interp(np.arange(n_timesteps), valid_idx, valid_vals)

                # 初始化单变量随机游走模型
                # 限制：transition_matrices 和 observation_matrices 设为 1
                kf = KalmanFilter(
                    transition_matrices=[1],
                    observation_matrices=[1],
                    initial_state_mean=temp_interp[0],
                    em_vars=[
                        "transition_covariance",
                        "observation_covariance",
                        "initial_state_mean",
                        "initial_state_covariance",
                    ]
                )
                
                # 运行 EM 算法估计噪声参数
                kf = kf.em(temp_interp.reshape(-1, 1), n_iter=5)
                
                # RTS 平滑
                smoothed_means, _ = kf.smooth(temp_interp.reshape(-1, 1))
                smoothed_values = smoothed_means.flatten()

                # 验证 Kalman 的结果是否有效（无 inf/nan 且没有产生极端溢出）
                if np.all(np.isfinite(smoothed_values)):
                    # 数值截断：防止 Kalman 在长距离预测时发散
                    # 将预测值限制在 [s_min, s_max] 范围内
                    series_filled[nan_mask] = np.clip(smoothed_values[nan_mask], s_min, s_max)
                    success_flag = True
                
            except Exception as e:
                if verbose:
                    print(f"  [Warning] 样本 {i} 特征 {d} Kalman 失败: {e}")

            # 情况 C: 回退到线性插值 (如果 Kalman 失败或产生无效值)
            if not success_flag:
                # 重新计算一次纯线性插值
                series_filled[nan_mask] = np.interp(
                    np.where(nan_mask)[0], np.where(~nan_mask)[0], valid_vals
                )

            # 情况 D: 安全防御（处理 np.interp 无法覆盖的情况，如边缘缺失且全局均值为 NaN）
            bad_mask = ~np.isfinite(series_filled)
            if np.any(bad_mask):
                series_filled[bad_mask] = global_feature_means[d]

            data_imputed[i, :, d] = series_filled

    # 4. 最终全数组加固，杜绝任何残余的 NaN/Inf
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed