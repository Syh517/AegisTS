import numpy as np
from sklearn.mixture import GaussianMixture
import warnings

def em_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, n_components=2
) -> np.ndarray:
    """
    基于 GMM 后验概率推断的自适应 EM 修复函数：
    1. 自动检测：计算样本在 GMM 下的得分，将低概率点判定为异常。
    2. 极大似然修复：利用 GMM 的各组件后验概率，将脏数据修复为所有高斯分量的加权期望。
    3. 鲁棒性：使用稳健的初始值，并结合特征间的空间相关性进行分维度修复。
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 展平以便全局学习分布形态
    flat_data = data_abnormal.reshape(-1, n_features)
    
    try:
        # --- 1. 拟合 GMM 模型 (M-Step) ---
        # 增加一个额外的组件或使用较低的 tol 以确保能容忍一定的初始噪声
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",  # 使用全协方差以捕捉特征间的空间相关性
            max_iter=100,
            random_state=42
        )
        gmm.fit(flat_data)
        
        # --- 2. 自动异常识别 (E-Step) ---
        # 计算每个点的对数似然得分
        scores = gmm.score_samples(flat_data)
        # 阈值判定：似然得分低于 5% 分位数的点被视为异常
        # 或者使用绝对阈值，取决于数据的噪声密度
        threshold = np.percentile(scores, 5) 
        is_dirty_flat = scores < threshold
        
        # --- 3. 极大似然修复 ---
        # 获取所有组件的均值和权重
        means = gmm.means_      # (n_components, n_features)
        weights = gmm.weights_  # (n_components,)
        covs = gmm.covariances_ # (n_components, n_features, n_features)

        # 预测所有点属于各组件的后验概率 (Responsibilities)
        responsibilities = gmm.predict_proba(flat_data) # (n_pts, n_components)

        for i in range(len(flat_data)):
            if not is_dirty_flat[i]:
                continue
            
            x = flat_data[i]
            # 识别具体哪些特征维度偏离太远 (超过各自维度的 sigma)
            # 使用全局标准差作为参考
            global_std = np.sqrt(np.diagonal(np.mean(covs, axis=0)))
            bad_mask = np.abs(x - np.average(means, axis=0, weights=weights)) > 3 * global_std
            
            if not np.any(bad_mask):
                # 如果整体得分低但找不到具体维度，则全量修复
                bad_mask = np.ones(n_features, dtype=bool)
            
            good_mask = ~bad_mask
            
            # 对每个高斯组件计算条件期望
            # E[x_bad] = sum( responsibility_k * E[x_bad | x_good, Component_k] )
            replacement = np.zeros(np.sum(bad_mask))
            
            for k in range(n_components):
                mu_k = means[k]
                cov_k = covs[k]
                
                res_k = responsibilities[i, k]
                
                if np.any(good_mask):
                    # 空间条件期望：利用第 k 个高斯分量的相关性进行推断
                    s12 = cov_k[np.ix_(bad_mask, good_mask)]
                    s22 = cov_k[np.ix_(good_mask, good_mask)] + np.eye(np.sum(good_mask)) * 1e-6
                    
                    cond_mu = mu_k[bad_mask] + s12 @ np.linalg.solve(s22, x[good_mask] - mu_k[good_mask])
                    replacement += res_k * cond_mu
                else:
                    # 如果没有好的维度，直接使用各组件均值的加权和
                    replacement += res_k * mu_k[bad_mask]
            
            flat_data[i, bad_mask] = replacement

    except Exception as e:
        # 如果 GMM 崩溃，退回到中值修复
        print(f"GMM Repair failed: {e}")
        median_val = np.median(flat_data, axis=0)
        flat_data[is_dirty_flat] = median_val

    return flat_data.reshape(n_samples, n_timestamps, n_features)