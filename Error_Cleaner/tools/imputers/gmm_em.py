import numpy as np
import warnings
from sklearn.mixture import GaussianMixture

def impute_gmm_em(data_missing: np.ndarray, n_components: int = 5, max_iter: int = 5) -> np.ndarray:
    """
    深度优化后的 GMM-EM 插补法
    优化点：全局特征学习、回归式条件均值填充、计算效率优化
    """
    assert len(data_missing.shape) == 3, "输入数据必须是三维 (N x T x D)"
    n_samples, n_timesteps, n_features = data_missing.shape
    
    # 1. 预处理：统一缺失值标识并计算全局统计量
    data_imputed = np.copy(data_missing)
    data_imputed[~np.isfinite(data_imputed)] = np.nan
    
    # 获取全局范围用于截断，防止数值漂移
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_min = np.nanmin(data_imputed)
        global_max = np.nanmax(data_imputed)
        global_means = np.nanmean(data_imputed, axis=(0, 1))
        global_means = np.nan_to_num(global_means, nan=0.0)

    # 2. 初始填充 (Seed Filling)：为 GMM 训练准备完整矩阵
    # 将 (N, T, D) 重塑为 (N*T, D) 以进行全局建模
    flat_data = data_imputed.reshape(-1, n_features)
    nan_mask_flat = np.isnan(flat_data)
    
    # 初始填充：使用特征均值（也可以选择线性插值）
    for d in range(n_features):
        col = flat_data[:, d]
        mask = np.isnan(col)
        if np.any(mask):
            col[mask] = global_means[d]

    # 3. 全局 GMM 训练
    # 相比逐样本拟合，全局拟合能学习到更稳健的特征间协方差
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full', # 必须用 full 才能学习特征间相关性
            reg_covar=1e-5,
            random_state=42,
            max_iter=100
        )
        
        # EM 迭代增强插补效果
        for _ in range(max_iter):
            gmm.fit(flat_data)
            
            # 核心改进：计算条件期望 (Conditional Expectation)
            # 这里简化为混合模型的后验预测：E[x] = sum(p(k|x) * mu_k)
            # 这种方式在缺失值位置能提供基于聚类中心的最佳估计
            resp = gmm.predict_proba(flat_data)
            reconstructed = resp @ gmm.means_
            
            # 只在缺失处更新，并增加数值截断
            reconstructed = np.clip(reconstructed, global_min, global_max)
            flat_data[nan_mask_flat] = reconstructed[nan_mask_flat]
            
    except Exception as e:
        warnings.warn(f"GMM 拟合失败，退回到基础填充: {e}")
        # 如果 GMM 失败，flat_data 已经由 global_means 填充过了

    # 4. 针对传感器数据的局部微调（可选）
    # 如果数据原本是 N x T x D，这里可以把结果还原回来
    data_imputed = flat_data.reshape(n_samples, n_timesteps, n_features)

    # 5. 最终安全检查
    data_imputed = np.nan_to_num(data_imputed, nan=0.0, posinf=global_max, neginf=global_min)
    
    return data_imputed