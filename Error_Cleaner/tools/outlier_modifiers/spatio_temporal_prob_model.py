import numpy as np

def spatio_temporal_prob_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None
) -> np.ndarray:
    """
    基于时空联合概率分布的自动修复模型：
    1. 自动检测：结合马氏距离（空间）和变化率（时间）自动判定脏数据。
    2. 空间 MLE：利用多元正态分布的条件均值，根据正常维度修复异常维度。
    3. 时间平滑：利用局部时间窗口的似然估计进行二次修正，消除阶跃。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 展平以便进行全局空间统计学习
    flat_data = data_abnormal.reshape(-1, n_features)
    
    # --- 1. 稳健空间特征估计 (M-Step) ---
    # 使用中位数和稳健协方差缩减（Shrinkage）
    mu_global = np.median(flat_data, axis=0)
    # 初始协方差：使用简单缩减防止奇异性
    cov_global = np.cov(flat_data, rowvar=False) + np.eye(n_features) * 1e-6
    
    try:
        inv_cov = np.linalg.inv(cov_global)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_global)

    # --- 2. 逐样本时空修复 ---
    for s in range(n_samples):
        sample_data = data_repaired[s]
        
        # 计算该样本内每个点的空间马氏距离
        diffs = sample_data - mu_global
        spatial_dists = np.sum((diffs @ inv_cov) * diffs, axis=1)
        
        # 计算时间跳变（一阶差分）
        temp_diffs = np.zeros(n_timestamps)
        temp_diffs[1:] = np.sum(np.abs(np.diff(sample_data, axis=0)), axis=1)
        
        # 自动识别脏点：空间距离过大 OR 时间跳变异常
        # 阈值：特征数 * 3 (空间) 和 均值跳变 3 倍 (时间)
        is_dirty_time = (spatial_dists > n_features * 3) | (temp_diffs > np.median(temp_diffs) * 5)
        
        if not np.any(is_dirty_time):
            continue

        for t in range(n_timestamps):
            if not is_dirty_time[t]:
                continue
                
            x_t = sample_data[t]
            # 识别具体的脏维度：偏离全局均值超过 2.5 sigma
            bad_mask = np.abs(x_t - mu_global) > 2.5 * np.sqrt(np.diag(cov_global))
            good_mask = ~bad_mask
            
            # --- 空间修复 (Conditioning) ---
            if np.any(good_mask) and np.any(bad_mask):
                s12 = cov_global[np.ix_(bad_mask, good_mask)]
                s22 = cov_global[np.ix_(good_mask, good_mask)]
                # MLE 空间推断值
                spatial_fill = mu_global[bad_mask] + s12 @ np.linalg.solve(s22, x_t[good_mask] - mu_global[good_mask])
            else:
                spatial_fill = mu_global[bad_mask]

            # --- 时间修正 (Local Temporal Smoothing) ---
            # 获取局部窗口内的正常点
            w = 2
            start, end = max(0, t-w), min(n_timestamps, t+w+1)
            # 排除掉当前已知的脏时间点
            local_context = sample_data[start:end][~is_dirty_time[start:end]]
            
            if local_context.size > 0:
                temporal_fill = np.mean(local_context, axis=0)[bad_mask]
                # 时空融合：70% 空间逻辑 + 30% 时间连续性
                data_repaired[s, t, bad_mask] = 0.7 * spatial_fill + 0.3 * temporal_fill
            else:
                data_repaired[s, t, bad_mask] = spatial_fill
                
    return data_repaired