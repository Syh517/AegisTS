import numpy as np
import warnings
from hmmlearn import hmm

def impute_hmm(data_missing: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    优化后的 HMM 插补法：增加协方差保护、数值截断和健壮性逻辑。
    
    :param data_missing: 待修复的 numpy 数组 (N x T x D)
    :param n_components: 隐藏状态数量
    :return: 修复后的 numpy 数组 (N x T x D)
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    # 1. 初始化与预处理
    data_imputed = np.copy(data_missing)
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 2. 计算全局特征均值 (D,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_feature_means = np.nanmean(data_imputed, axis=(0, 1))
        global_feature_means = np.nan_to_num(global_feature_means, nan=0.0)

    # 3. 逐样本处理
    for i in range(n_samples):
        sample = data_imputed[i]
        nan_mask = np.isnan(sample)
        
        if not np.any(nan_mask):
            continue

        # 记录样本有效范围用于 Clipping
        sample_finite = sample[~nan_mask]
        if sample_finite.size > 0:
            s_min, s_max = np.min(sample_finite), np.max(sample_finite)
        else:
            data_imputed[i] = np.tile(global_feature_means, (n_timesteps, 1))
            continue
            
        # 情况 A: 初始种子填充 (Seeding)
        seeded_sample = sample.copy()
        for d in range(n_features):
            col = seeded_sample[:, d]
            c_nan = np.isnan(col)
            if np.any(c_nan):
                v_idx = np.where(~c_nan)[0]
                if v_idx.size > 0:
                    col[c_nan] = np.interp(np.where(c_nan)[0], v_idx, col[v_idx])
                else:
                    col[:] = global_feature_means[d]
        
        # 情况 B: 尝试 HMM 建模
        try:
            # 增加 min_covar 防止协方差矩阵奇异导致 NaN
            model = hmm.GaussianHMM(
                n_components=n_components, 
                covariance_type="diag", 
                min_covar=1e-3, 
                n_iter=15, 
                random_state=42
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(seeded_sample)
                
                # Viterbi 解码
                # logprob 是对数概率，如果为 -inf 说明模型失效
                logprob, state_sequence = model.decode(seeded_sample, algorithm="viterbi")
                
            if np.isfinite(logprob):
                state_means = model.means_
                
                # 检查学习到的均值矩阵是否有效
                if np.all(np.isfinite(state_means)):
                    # 根据状态序列生成重构值
                    reconstructed = state_means[state_sequence]
                    
                    # 关键：数值截断，确保 HMM 状态均值不超出实际观测物理范围
                    reconstructed = np.clip(reconstructed, s_min, s_max)
                    
                    # 填充
                    seeded_sample[nan_mask] = reconstructed[nan_mask]
        
        except Exception:
            # 失败则保留 seeded_sample (即线性插值的结果)
            pass

        # 情况 C: 样本级安全性加固
        final_bad_mask = ~np.isfinite(seeded_sample)
        if np.any(final_bad_mask):
            # 逐列检查并用全局均值修复
            for d in range(n_features):
                col_bad = final_bad_mask[:, d]
                if np.any(col_bad):
                    seeded_sample[col_bad, d] = global_feature_means[d]
                
        data_imputed[i] = seeded_sample

    # 4. 最终全数组加固
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data_imputed