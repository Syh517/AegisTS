import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

def clustering_kmeans_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, n_clusters=5, contamination=0.05
) -> np.ndarray:
    """
    自适应 KMeans 修复函数（不依赖 label）：
    1. 自动识别：利用孤立森林 (Isolation Forest) 自动识别未知的脏数据。
    2. 鲁棒标准化：仅利用识别出的“正常点”进行标准化拟合。
    3. 模式提取：对正常点进行 KMeans 聚类提取典型模式。
    4. 距离映射：将识别出的脏数据和 NaN 点映射到最相似的正常簇中心。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    scaler = StandardScaler()
    # 用于检测前的临时均值填充
    pre_imputer = SimpleImputer(strategy='mean')
    
    for s in range(n_samples):
        current_sample = data_abnormal[s]
        
        # 0. 基础检查：如果全是 NaN 则跳过
        if np.isnan(current_sample).all():
            continue

        # --- Step 1: 自动异常检测 (取代 label) ---
        try:
            # 临时填补 NaN 以便运行检测算法
            data_temp = pre_imputer.fit_transform(current_sample)
            X_temp_scaled = scaler.fit_transform(data_temp)
            
            # 使用孤立森林识别异常 (1:正常, -1:异常)
            iso = IsolationForest(contamination=contamination, random_state=42)
            det_labels = iso.fit_predict(X_temp_scaled)
        except:
            continue

        # 构建掩码：NaN 点 或 被检测为离群点的点
        mask_is_nan = np.isnan(current_sample).any(axis=1)
        mask_abnormal = (det_labels == -1) | mask_is_nan
        mask_normal = ~mask_abnormal
        
        # 如果正常点太少，无法建立聚类模型
        if np.sum(mask_normal) < n_clusters:
            # 兜底逻辑：用所有点的全局均值修复
            col_mean = np.nanmean(current_sample, axis=0)
            data_repaired[s, mask_abnormal] = col_mean
            continue

        # --- Step 2: 建立聚类模型 ---
        normal_X = current_sample[mask_normal]
        
        # 如果正常点里还有 NaN（极少见），补齐
        if np.isnan(normal_X).any():
            normal_X = pre_imputer.fit_transform(normal_X)

        # 仅用正常点拟合标准化器
        normal_X_scaled = scaler.fit_transform(normal_X)
        
        # 在标准化空间进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(normal_X_scaled)

        # --- Step 3: 映射与修复 ---
        # 准备所有待修复的点（脏数据和 NaN）
        abnormal_X = data_temp[mask_abnormal] # 使用 data_temp 保证没有 NaN
        abnormal_X_scaled = scaler.transform(abnormal_X)
        
        # 寻找最近的中心索引
        closest_ids = pairwise_distances_argmin(abnormal_X_scaled, kmeans.cluster_centers_)
        
        # 获取原始空间的中心点并回填
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        data_repaired[s, mask_abnormal] = centers_original[closest_ids]

    return data_repaired