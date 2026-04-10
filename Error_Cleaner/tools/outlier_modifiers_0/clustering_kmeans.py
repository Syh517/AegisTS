import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from sklearn.impute import KNNImputer

def clustering_kmeans_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_clusters=5
) -> np.ndarray:
    """
    优化后的 KMeans 修复函数：
    1. 逻辑纠正：严格执行 [标准化 -> 聚类 -> 映射] 流程。
    2. 效率优化：批量处理异常点，减少重复的 transform 运算。
    3. 健壮性：处理正常点过少或特征全零导致的标准化异常。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 预重用对象以减少开销
    scaler = StandardScaler()
    
    for s in range(n_samples):
        mask_normal = (label[s] == 0)
        mask_abnormal = (label[s] == 1)
        
        if not np.any(mask_normal) or not np.any(mask_abnormal):
            continue

        normal_X = data_abnormal[s, mask_normal]
        
        # Step 1: 仅对正常数据进行缺失值预处理
        # 如果正常点没缺失值，不需要管异常点的缺失值（因为它们最终会被中心值替换）
        if np.isnan(normal_X).any():
            n_knn = min(5, max(1, normal_X.shape[0]))
            imputer = KNNImputer(n_neighbors=n_knn)
            normal_X = imputer.fit_transform(normal_X)

        # Step 2: 严格逻辑 - 先标准化
        try:
            # 仅用正常点拟合，确保异常点不会污染标准差
            normal_X_scaled = scaler.fit_transform(normal_X)
        except ValueError: # 处理样本过少或方差为0的情况
            data_repaired[s, mask_abnormal] = np.nanmean(normal_X, axis=0)
            continue

        # Step 3: 在标准化空间进行聚类
        eff_clusters = min(n_clusters, normal_X.shape[0])
        if eff_clusters < 1: continue
        
        kmeans = KMeans(n_clusters=eff_clusters, random_state=42, n_init='auto')
        kmeans.fit(normal_X_scaled)
        
        # Step 4: 修复异常点
        # 将异常点映射到相同的标准空间
        abnormal_X = data_abnormal[s, mask_abnormal]
        # 如果异常点包含 NaN，先用正常点均值简单填充，以便计算到簇中心的距离
        if np.isnan(abnormal_X).any():
            col_mean = np.nanmean(normal_X, axis=0)
            inds = np.where(np.isnan(abnormal_X))
            abnormal_X[inds] = np.take(col_mean, inds[1])
            
        abnormal_X_scaled = scaler.transform(abnormal_X)
        
        # 在标准空间寻找最近的中心索引
        closest_ids = pairwise_distances_argmin(abnormal_X_scaled, kmeans.cluster_centers_)
        
        # Step 5: 还原到原始空间
        # 直接使用原始空间的中心点（通过逆缩放获得，更精确）
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        data_repaired[s, mask_abnormal] = centers_original[closest_ids]

    return data_repaired