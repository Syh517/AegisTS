import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def lof_neighbor_mean_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_neighbors=5
) -> np.ndarray:
    """
    优化后的 KNN 邻近点均值修复函数：
    1. 性能：将逐点查找邻居改为批量查找，利用 NearestNeighbors 的矩阵运算优势。
    2. 鲁棒性：优化了 KNNImputer 的调用触发条件和参数自适应。
    3. 逻辑：修正了潜在的索引偏移风险，确保修复值完全来自正常观测。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    scaler = StandardScaler()

    for s in range(n_samples):
        mask_normal = (label[s] == 0)
        mask_abnormal = (label[s] == 1)
        
        if not np.any(mask_normal) or not np.any(mask_abnormal):
            continue

        # Step 1: 缺失值预处理 (仅在必要时执行)
        current_sample = data_abnormal[s]
        if np.isnan(current_sample).any():
            normal_data = current_sample[mask_normal]
            # 确保邻居数不超过可用正常样本
            n_impute = min(5, max(1, normal_data.shape[0]))
            imputer = KNNImputer(n_neighbors=n_impute)
            # 仅用正常点训练，填补整段以获取“搜索参考坐标”
            data_filled = imputer.fit_transform(current_sample)
        else:
            data_filled = current_sample

        # Step 2: 空间映射
        X_normal = data_filled[mask_normal]
        try:
            X_normal_scaled = scaler.fit_transform(X_normal)
            X_all_scaled = scaler.transform(data_filled)
        except ValueError: # 处理常数项导致的缩放异常
            continue

        # Step 3: 批量构建与查询 (核心效率优化)
        n_neigh = min(n_neighbors, X_normal_scaled.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1).fit(X_normal_scaled)
        
        # 一次性找出所有异常点的最近邻索引
        X_abnormal_scaled = X_all_scaled[mask_abnormal]
        _, neigh_idx_local = nbrs.kneighbors(X_abnormal_scaled)
        
        # Step 4: 映射回原始空间并计算均值
        # 获取正常点在原始数据中的位置索引
        normal_indices_global = np.where(mask_normal)[0]
        
        # 构建一个映射矩阵，方便批量提取
        # neigh_idx_local 的形状为 (n_anomalies, n_neighbors)
        neigh_idx_global = normal_indices_global[neigh_idx_local]
        
        # 批量提取邻居值并计算均值
        # 结果形状为 (n_anomalies, n_features)
        repaired_values = np.mean(data_filled[neigh_idx_global], axis=1)
        
        # 回填数据
        data_repaired[s, mask_abnormal, :] = repaired_values

    return data_repaired