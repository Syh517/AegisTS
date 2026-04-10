import numpy as np
from sklearn.cluster import KMeans

def relationship_network_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_clusters=3
):
    """
    优化后的关系依赖网络修复函数：
    1. 效率优化：移除冗余的 KNNImputer，直接利用 KMeans 建立特征空间依赖。
    2. 逻辑增强：引入“模式匹配”逻辑，寻找异常点前后的最近正常状态。
    3. 性能：减少对象初始化，优化矩阵赋值。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for s in range(n_samples):
        is_normal = (label[s] == 0)
        is_abnormal = (label[s] == 1)
        
        if not np.any(is_normal) or not np.any(is_abnormal):
            continue
            
        normal_data = data_abnormal[s, is_normal, :]
        
        # 基础校验：如果正常点还没聚类中心多，直接降级为均值填充
        if len(normal_data) < n_clusters:
            data_repaired[s, is_abnormal, :] = np.mean(normal_data, axis=0)
            continue

        try:
            # 1. 学习空间关系依赖（聚类提取典型模式）
            # n_init='auto' 在新版 sklearn 中更快
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(normal_data)
            centers = kmeans.cluster_centers_
            
            # 2. 修复逻辑优化：
            # 为了避免单纯用中心值覆盖导致的“阶跃”现象，
            # 我们寻找异常点前后的正常观测值，看它们属于哪个 Cluster
            # 并根据距离进行加权修复
            bad_indices = np.where(is_abnormal)[0]
            
            # 预计算所有时间点的类别归属（对于异常点，基于周围正常点推测）
            # 这种方法比原代码先用 KNN 填再 predict 要快得多
            for idx in bad_indices:
                # 寻找最近的正常点参考
                # 寻找距离 idx 最近的 is_normal 为 True 的索引
                dist_to_normal = np.abs(np.where(is_normal)[0] - idx)
                nearest_normal_idx = np.where(is_normal)[0][np.argmin(dist_to_normal)]
                
                # 获取最近正常点的典型模式
                nearest_val = data_abnormal[s, nearest_normal_idx, :].reshape(1, -1)
                cluster_label = kmeans.predict(nearest_val)[0]
                
                # 修复：使用对应的聚类中心
                data_repaired[s, idx, :] = centers[cluster_label]
                
                # 局部平滑：防止修复点与最近正常点差异过大
                data_repaired[s, idx, :] = 0.5 * data_repaired[s, idx, :] + 0.5 * nearest_val

        except Exception:
            data_repaired[s, is_abnormal, :] = np.mean(normal_data, axis=0)

    return data_repaired