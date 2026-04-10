import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

def lof_neighbor_mean_repair(
    data_abnormal: np.ndarray, label=None, n_neighbors=5, contamination=0.2
) -> np.ndarray:
    """
    自适应 KNN 邻近点修复函数（不依赖外部 label）：
    1. 自动检测：使用 IsolationForest 自动识别未标记的脏数据。
    2. 统一处理：合并 NaN 和检测到的离群点，作为修复目标。
    3. 邻域修复：基于“干净模式”点集，批量计算最近邻均值进行修复。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    scaler = StandardScaler()
    # 用于检测前的临时填补
    pre_imputer = SimpleImputer(strategy='median')

    for s in range(n_samples):
        current_sample = data_abnormal[s] # (n_timestamps, n_features)
        
        # 0. 基础检查
        if np.isnan(current_sample).all():
            continue

        # --- Step 1: 自动异常检测 (取代 label) ---
        # 临时填补 NaN 以便运行检测算法
        try:
            data_temp = pre_imputer.fit_transform(current_sample)
            X_scaled = scaler.fit_transform(data_temp)
        except:
            continue

        # 使用孤立森林检测脏数据 (1为正常, -1为异常)
        iso = IsolationForest(contamination=contamination, random_state=42)
        det_labels = iso.fit_predict(X_scaled)

        # 构建掩码：NaN点 OR 检测出的离群点
        mask_is_nan = np.isnan(current_sample).any(axis=1)
        mask_is_outlier = (det_labels == -1)
        mask_abnormal = mask_is_nan | mask_is_outlier
        mask_normal = ~mask_abnormal

        # 如果没有异常需要修复，或没有正常参考点，则跳过
        if not np.any(mask_abnormal) or not np.any(mask_normal):
            continue

        # --- Step 2: 准备搜索空间 ---
        # 我们只从“确定干净”的点中寻找邻居
        X_normal_scaled = X_scaled[mask_normal]
        
        # --- Step 3: 批量查找最近邻 ---
        # 确保邻居数不超过可用正常样本数
        actual_neighbors = min(n_neighbors, X_normal_scaled.shape[0])
        nbrs = NearestNeighbors(n_neighbors=actual_neighbors, n_jobs=-1).fit(X_normal_scaled)
        
        # 查找所有异常点的最近邻索引
        X_abnormal_scaled = X_scaled[mask_abnormal]
        _, neigh_idx_local = nbrs.kneighbors(X_abnormal_scaled)
        
        # --- Step 4: 映射与均值计算 ---
        # 将局部索引转换为该样本内的全局时间戳索引
        normal_indices_global = np.where(mask_normal)[0]
        neigh_idx_global = normal_indices_global[neigh_idx_local]
        
        # 使用原始（或初步填补后的）干净数据计算均值
        # 注意：使用 data_temp 确保即使邻居中有被填补的点也能参与运算
        repaired_values = np.mean(data_temp[neigh_idx_global], axis=1)
        
        # 回填数据
        data_repaired[s, mask_abnormal, :] = repaired_values

    return data_repaired