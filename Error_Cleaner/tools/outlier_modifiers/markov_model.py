import numpy as np

def markov_model_repair(data_abnormal: np.ndarray, label: np.ndarray = None) -> np.ndarray:
    """
    基于马尔可夫转移概率的自适应 MLE 修复函数：
    1. 移除对 label 的依赖，通过转移似然度自动定位异常。
    2. 统计模型：假设相邻时间步的差分服从正态分布 Delta ~ N(mu_delta, sigma_delta)。
    3. 极大似然路径：对于识别出的异常段，利用双向马尔可夫链进行最大概率推断。
    """
    n_samples, n_timestamps, n_features = data_abnormal.shape
    data_repaired = data_abnormal.copy().astype(float)

    for i in range(n_samples):
        for j in range(n_features):
            series = data_repaired[i, :, j]
            
            # --- 1. 学习转移概率分布 (MLE) ---
            # 计算一阶差分
            diffs = np.diff(series)
            # 使用稳健统计（中位数和 MAD）来估计正常状态下的转移均值和标准差
            mu_delta = np.median(diffs)
            sigma_delta = np.median(np.abs(diffs - mu_delta)) * 1.4826 + 1e-6
            
            # --- 2. 自动异常识别 ---
            # 计算每个点相对于马尔可夫转移的似然度
            # 如果 |diff - mu| > 3 * sigma，则认为发生了非自然的“跳变”
            z_scores = np.abs(diffs - mu_delta) / sigma_delta
            # 标记异常转移点 (注意：diffs 比 series 少一个元素)
            bad_transitions = z_scores > 3.0
            
            if not np.any(bad_transitions):
                continue
                
            # --- 3. 极大似然路径修复 ---
            # 我们将连续的异常段找出来进行统一修复
            # 构建一个内部 label (1 代表异常)
            internal_label = np.zeros(n_timestamps, dtype=bool)
            for t, is_bad in enumerate(bad_transitions):
                if is_bad:
                    internal_label[t+1] = True # 转移异常通常意味着目标点是脏点

            # 修复逻辑：双向马尔可夫融合
            # 找到正常点的索引
            normal_indices = np.where(~internal_label)[0]
            if len(normal_indices) < 2:
                # 全局漂移过大时，使用全局均值
                series[internal_label] = np.mean(series)
                continue

            # 使用线性插值作为基准，叠加载有状态转移信息的趋势
            # 这等价于在给定前后端点约束下，最大化 P(X_t | X_{t-1}, X_{t+1})
            bad_indices = np.where(internal_label)[0]
            
            # 分段处理连续的异常块
            groups = np.split(bad_indices, np.where(np.diff(bad_indices) > 1)[0] + 1)
            
            for group in groups:
                if len(group) == 0: continue
                
                # 确定该段异常的前后“锚点”
                start_idx = group[0] - 1
                end_idx = group[-1] + 1
                
                # 处理边界情况
                t_start = max(0, start_idx)
                t_end = min(n_timestamps - 1, end_idx)
                
                val_start = series[t_start]
                val_end = series[t_end]
                
                # 在起始点和结束点之间，按马尔可夫步长 mu_delta 进行插值
                # 这种方法保证了修复段既平滑，又符合该特征的历史转移趋势
                dist = t_end - t_start
                for step, t_idx in enumerate(group):
                    # 权重计算（距离起始点越近，受 start 影响越大）
                    weight_end = (t_idx - t_start) / dist
                    # 融合预测：(起点推导值) + (终点反向推导值)
                    # 这是马尔可夫平滑（Markov Smoothing）的简化实现
                    pred_from_start = val_start + (t_idx - t_start) * mu_delta
                    pred_from_end = val_end - (t_end - t_idx) * mu_delta
                    
                    series[t_idx] = (1 - weight_end) * pred_from_start + weight_end * pred_from_end
            
            data_repaired[i, :, j] = series

    return data_repaired