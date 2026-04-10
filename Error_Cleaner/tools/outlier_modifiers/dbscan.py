import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer

def dbscan_repair(
    data_abnormal: np.ndarray, label=None, eps=0.5, min_samples=5
) -> np.ndarray:
    """
    自适应 DBSCAN 修复函数（不依赖外部 label）：
    1. 自动识别：利用 DBSCAN 的噪声检测 (-1) 自动定位脏数据。
    2. 统一修复：将 NaN 和噪声点视为同一类待修复目标。
    3. 鲁棒统计：使用中位数 (Median) 替代均值，增强对残余噪声的抵抗力。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    scaler = StandardScaler()
    # 使用中位数填充 NaN 作为 DBSCAN 的预处理输入（防止 NaN 导致无法聚类）
    pre_imputer = SimpleImputer(strategy='median')

    for s in range(n_samples):
        current_sample = data_abnormal[s] # 形状 (n_timestamps, n_features)
        
        # --- Step 1: 预处理 NaN ---
        # 如果整行都是 NaN，无法处理，跳过
        if np.isnan(current_sample).all():
            continue
            
        # 临时填补 NaN 以便进行 DBSCAN 聚类
        try:
            data_temp_filled = pre_imputer.fit_transform(current_sample)
        except ValueError:
            continue

        # --- Step 2: 标准化与聚类 ---
        try:
            X_scaled = scaler.fit_transform(data_temp_filled)
        except:
            continue

        # 使用 DBSCAN 识别“正常模式”
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        db_labels = db.labels_

        # 定义掩码：噪声点 (-1) 和 原始数据中的 NaN 都是待修复点
        mask_is_nan = np.isnan(current_sample).any(axis=1)
        mask_is_noise = (db_labels == -1)
        mask_to_repair = mask_is_nan | mask_is_noise
        mask_is_clean = ~mask_to_repair

        # 如果没有干净的点，或者全是干净点，无需修复
        if not np.any(mask_is_clean) or not np.any(mask_to_repair):
            continue

        # --- Step 3: 提取有效簇中心 ---
        unique_clusters = np.unique(db_labels[db_labels != -1])
        
        if len(unique_clusters) == 0:
            # 极端情况：没有形成簇，使用全局干净数据的中位数修复
            clean_median = np.nanmedian(current_sample[mask_is_clean], axis=0)
            data_repaired[s, mask_to_repair, :] = clean_median
            continue

        # 计算各个簇的鲁棒中心（中位数）
        # 注意：这里在原始空间计算，避免反复逆缩放带来的误差
        centers = []
        centers_scaled = []
        for c in unique_clusters:
            cluster_mask = (db_labels == c)
            # 簇中心：在缩放空间用于距离计算，在原始空间用于填补
            centers.append(np.median(current_sample[cluster_mask & ~mask_is_nan], axis=0))
            centers_scaled.append(X_scaled[cluster_mask].mean(axis=0))

        centers = np.array(centers)
        centers_scaled = np.array(centers_scaled)

        # --- Step 4: 匹配与修复 ---
        X_to_repair_scaled = X_scaled[mask_to_repair]
        
        # 寻找距离待修复点最近的正常簇中心
        nearest_indices, _ = pairwise_distances_argmin_min(X_to_repair_scaled, centers_scaled)
        
        # 执行修复
        data_repaired[s, mask_to_repair, :] = centers[nearest_indices]

    return data_repaired