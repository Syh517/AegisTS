import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import KNNImputer

def dbscan_repair(
    data_abnormal: np.ndarray, label: np.ndarray, eps=0.5, min_samples=5
) -> np.ndarray:
    """
    优化后的 DBSCAN 修复函数：
    1. 向量化处理：一次性计算所有异常点到簇中心的距离，移除内部循环。
    2. 维度保护：仅针对 label 标记的异常点进行修复，保留其空间位置参考。
    3. 异常处理：增加了对全 NaN 或全异常样本的健壮性检查。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 预初始化常用的工具
    scaler = StandardScaler()

    for s in range(n_samples):
        mask_normal = (label[s] == 0)
        mask_abnormal = (label[s] == 1)
        
        if not np.any(mask_normal):
            continue

        # Step 1: 处理缺失值 (NaN)
        # 仅当当前样本包含 NaN 时才调用昂贵的 KNNImputer
        current_sample = data_abnormal[s]
        if np.isnan(current_sample).any():
            normal_data_for_impute = current_sample[mask_normal]
            # 这里的 n_neighbors 不能超过正常样本数
            n_knn = min(5, max(1, normal_data_for_impute.shape[0]))
            imputer = KNNImputer(n_neighbors=n_knn)
            # 拟合正常数据填补整体
            data_filled = imputer.fit_transform(current_sample)
        else:
            data_filled = current_sample

        # Step 2: 标准化 (DBSCAN 对距离敏感，必须标准化)
        try:
            X_normal = data_filled[mask_normal]
            X_normal_scaled = scaler.fit_transform(X_normal)
            X_all_scaled = scaler.transform(data_filled)
        except ValueError: # 防止数据全为常数导致的除零
            continue

        # Step 3: DBSCAN 聚类
        # 实际 min_samples 不能大于正常点数
        effective_min_samples = min(min_samples, max(1, X_normal_scaled.shape[0]))
        db = DBSCAN(eps=eps, min_samples=effective_min_samples).fit(X_normal_scaled)
        db_labels = db.labels_

        # Step 4: 计算簇中心
        unique_clusters = np.unique(db_labels[db_labels != -1])
        
        if len(unique_clusters) == 0:
            # 如果没有形成任何簇，退化为使用正常点的全局均值修复
            fill_val = np.mean(X_normal, axis=0)
            data_repaired[s, mask_abnormal, :] = fill_val
            continue

        # 向量化计算中心
        centers_scaled = np.array([X_normal_scaled[db_labels == c].mean(axis=0) for c in unique_clusters])

        # Step 5: 向量化寻找最近中心并修复
        X_abnormal_scaled = X_all_scaled[mask_abnormal]
        
        # 一次性找到所有异常点对应的最近中心索引
        nearest_indices, _ = pairwise_distances_argmin_min(X_abnormal_scaled, centers_scaled)
        
        # 获取对应的中心点并逆标准化
        repaired_scaled = centers_scaled[nearest_indices]
        repaired_original = scaler.inverse_transform(repaired_scaled)
        
        # 更新修复数据
        data_repaired[s, mask_abnormal, :] = repaired_original

    return data_repaired