import numpy as np
from sklearn.linear_model import BayesianRidge

def bayesian_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None
) -> np.ndarray:
    """
    基于特征间贝叶斯概率推理的自动修复工具：
    1. 放弃时间轴回归，转向跨特征联合推断（利用特征相关性）。
    2. 自动异常检测：利用贝叶斯预测的不确定度（Sigma）识别离群点。
    3. 移除对输入 label 的强制依赖，实现闭环检测与修复。
    """
    n_samples, n_timestamps, n_features = data_abnormal.shape
    data_repaired = data_abnormal.copy().astype(float)
    
    # 展平数据以便全局学习特征间的贝叶斯关系
    flat_data = data_abnormal.reshape(-1, n_features)
    repaired_flat = flat_data.copy()

    # 对每一个特征，利用其他特征作为自变量建立贝叶斯模型
    # 例如：Feature_0 = w1*Feature_1 + w2*Feature_2 + ... + bias
    for j in range(n_features):
        target = flat_data[:, j]
        # 选择除当前特征外的所有其他特征作为预测源
        selectors = [idx for idx in range(n_features) if idx != j]
        features_other = flat_data[:, selectors]
        
        # 1. 初步筛选：利用鲁棒统计剔除明显的训练噪声，建立干净的基准模型
        # 使用中位数绝对偏差 (MAD) 快速过滤掉极其严重的离群点
        median = np.median(target)
        mad = np.median(np.abs(target - median))
        clean_train_mask = np.abs(target - median) < 3 * (mad + 1e-6)
        
        if np.sum(clean_train_mask) < 10: # 如果数据太脏，保底使用全部数据
            clean_train_mask = np.ones(len(target), dtype=bool)
            
        model = BayesianRidge()
        model.fit(features_other[clean_train_mask], target[clean_train_mask])
        
        # 2. 预测与概率化异常检测
        # 返回均值和标准差 (return_std=True 是贝叶斯模型的精髓)
        y_mu, y_std = model.predict(features_other, return_std=True)
        
        # 3. 识别脏数据：如果实际值偏离预测均值超过 3 倍标准差，认定为脏
        # 这种方法比硬性的阈值更科学，因为它考虑了模型本身对该点的信心
        is_dirty = np.abs(target - y_mu) > 3 * y_std
        
        # 4. 极大似然修复：将脏数据替换为模型预测的最优值（均值）
        repaired_flat[is_dirty, j] = y_mu[is_dirty]

    return repaired_flat.reshape(n_samples, n_timestamps, n_features)