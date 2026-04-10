import numpy as np
from sklearn.cluster import KMeans

def relationship_network_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, n_clusters=3
) -> np.ndarray:
    """
    基于 KMeans 模式识别的自适应关系修复：
    1. 移除 label 依赖：通过点到聚类中心的距离自动识别“偏离模式”的异常点。
    2. 模式投影：利用 MLE 原则，将异常点投影到特征空间中似然度最高（最近）的典型模式中心。
    3. 局部拓扑保护：结合邻域信息进行加权，保持时序轨迹的连贯。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 展平数据以学习特征空间的全局拓扑
    flat_data = data_abnormal.reshape(-1, n_features)
    
    try:
        # --- 1. 学习特征空间关系模式 (MLE 模型初始化) ---
        # 假设数据由几个主要运行模态组成
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(flat_data)
        centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
        # --- 2. 自动异常检测 (基于距离似然) ---
        # 计算每个点到其分配中心的距离
        distances = np.linalg.norm(flat_data - centers[cluster_labels], axis=1)
        # 计算每个类簇内部的平均距离和标准差
        # 阈值：类簇均值 + 3 * 标准差
        thresholds = np.zeros(n_clusters)
        for k in range(n_clusters):
            cluster_dist = distances[cluster_labels == k]
            if len(cluster_dist) > 0:
                thresholds[k] = np.mean(cluster_dist) + 3 * np.std(cluster_dist)
            else:
                thresholds[k] = np.inf
        
        # 判定脏数据
        is_dirty_flat = distances > thresholds[cluster_labels]
        
        # --- 3. 模式匹配修复 ---
        for i in range(len(flat_data)):
            if not is_dirty_flat[i]:
                continue
                
            x_dirty = flat_data[i]
            # 找到最接近的聚类中心作为 MLE 修复目标
            # (即在该点位置，哪个聚类模态出现的概率最大)
            best_cluster_idx = cluster_labels[i] 
            target_center = centers[best_cluster_idx]
            
            # 考虑时间平滑：引入前后相邻点的状态
            # 将 flat 索引转回 (s, t) 索引
            s_idx = i // n_timestamps
            t_idx = i % n_timestamps
            
            # 获取局部窗口内的正常点均值
            w = 2
            t_start, t_end = max(0, t_idx-w), min(n_timestamps, t_idx+w+1)
            # 这里的 mask 需要从 is_dirty_flat 还原
            local_window = data_abnormal[s_idx, t_start:t_end, :]
            local_dirty_mask = is_dirty_flat.reshape(n_samples, n_timestamps)[s_idx, t_start:t_end]
            
            local_normal = local_window[~local_dirty_mask]
            
            if len(local_normal) > 0:
                local_mu = np.mean(local_normal, axis=0)
                # 融合修复：60% 空间模式中心 + 40% 时间局部趋势
                flat_data[i] = 0.6 * target_center + 0.4 * local_mu
            else:
                flat_data[i] = target_center

    except Exception:
        # 兜底：全局均值
        mu_global = np.nanmean(flat_data, axis=0)
        flat_data[np.isnan(flat_data)] = mu_global

    return flat_data.reshape(n_samples, n_timestamps, n_features)