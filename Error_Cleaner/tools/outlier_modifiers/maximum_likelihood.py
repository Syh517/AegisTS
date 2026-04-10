import numpy as np

def maximum_likelihood_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None
) -> np.ndarray:
    """
    自适应 MLE 修复：不依赖外部 label，自动识别异常并利用多元统计关系修复。
    """
    n_samples, n_timestamps, n_features = data_abnormal.shape
    # 展平以便进行全局统计推断
    data_flat = data_abnormal.reshape(-1, n_features).copy()
    
    # --- 1. 稳健参数估计 (M-Step) ---
    # 为了防止脏数据污染均值和协方差，先用中位数和截断法做初步估计
    mu = np.median(data_flat, axis=0)
    # 计算协方差时，排除掉极端离群点的影响
    sigma = np.cov(data_flat, rowvar=False) + np.eye(n_features) * 1e-6
    
    # 迭代优化分布参数（简化的 EM 过程）
    for _ in range(2):
        try:
            inv_sigma = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            inv_sigma = np.linalg.pinv(sigma)
            
        # 计算每个点相对于当前分布的马氏距离
        diff = data_flat - mu
        # (N, features) @ (features, features) -> (N, features)
        md_squared = np.sum((diff @ inv_sigma) * diff, axis=1)
        
        # 找出相对“干净”的点（马氏距离在合理范围内，如卡方分布 97.5% 分位数）
        # 对于多变量数据，阈值通常设为特征数的 3 倍左右
        threshold = n_features * 3 
        clean_mask = md_squared < threshold
        
        if np.sum(clean_mask) > n_features:
            mu = np.mean(data_flat[clean_mask], axis=0)
            sigma = np.cov(data_flat[clean_mask], rowvar=False) + np.eye(n_features) * 1e-6

    # --- 2. 自动识别与条件修复 (E-Step) ---
    data_repaired_flat = data_flat.copy()
    
    for i in range(len(data_flat)):
        x = data_flat[i]
        diff = x - mu
        dist = diff.T @ inv_sigma @ diff
        
        # 如果该时间点整体偏离分布
        if dist > threshold:
            # 识别具体是哪些维度出了问题 (偏离均值超过 2 倍标准差)
            std_dev = np.sqrt(np.diag(sigma))
            bad_dims = np.abs(x - mu) > 2 * std_dev
            good_dims = ~bad_dims
            
            # 如果还有好的维度，就根据好的维度预测坏的维度
            if np.any(good_dims) and np.any(bad_dims):
                s12 = sigma[np.ix_(bad_dims, good_dims)]
                s22 = sigma[np.ix_(good_dims, good_dims)]
                
                # 条件期望公式: E(x_bad | x_good) = mu_bad + S12 * inv(S22) * (x_good - mu_good)
                prediction = mu[bad_dims] + s12 @ np.linalg.solve(s22, (x[good_dims] - mu[good_dims]))
                data_repaired_flat[i, bad_dims] = prediction
            elif not np.any(good_dims):
                # 如果全坏了，保守起见填入均值
                data_repaired_flat[i, :] = mu

    return data_repaired_flat.reshape(n_samples, n_timestamps, n_features)