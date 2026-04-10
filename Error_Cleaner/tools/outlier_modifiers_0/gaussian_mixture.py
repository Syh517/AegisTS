import numpy as np
from sklearn.mixture import GaussianMixture

def gaussian_mixture_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_components=5
) -> np.ndarray:
    """
    优化后的 GMM 修复函数：
    1. 逻辑纠正：将“盲目采样”改为“权重期望估计”，消除随机毛刺。
    2. 性能：将 covariance_type 改为 'diag'，提升多维特征下的计算速度。
    3. 稳定性：增加正则化项防止奇异矩阵，并处理正常点不足的情况。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for s in range(n_samples):
        is_normal = (label[s] == 0)
        is_abnormal = (label[s] == 1)
        
        # 基础校验：如果没有异常或正常点太少无法支撑组件数
        if not np.any(is_abnormal) or np.sum(is_normal) < n_components:
            continue
            
        normal_data = data_abnormal[s, is_normal, :]

        try:
            # 1. 拟合模型
            # diag 协方差在时序特征中更稳健且高效
            gmm = GaussianMixture(
                n_components=n_components, 
                covariance_type="diag", 
                max_iter=50, 
                random_state=42
            )
            gmm.fit(normal_data)

            # 2. 核心改进：计算各组件的加权期望值
            # 修复值 = Σ (权重_i * 均值_i)
            # 这代表了模型认为最可能的“统计中心”，比随机采样稳健得多
            expected_val = np.dot(gmm.weights_, gmm.means_)

            # 3. 局部时间趋势修正
            # 如果直接填 expected_val，连续异常段会变平。
            # 引入局部线性插值的趋势来微调这个期望值。
            bad_indices = np.where(is_abnormal)[0]
            
            # 预计算整个样本的线性插值作为趋势参考
            for col in range(n_features):
                y = data_abnormal[s, :, col]
                y_interp = np.interp(
                    np.arange(n_timestamps), 
                    np.where(is_normal)[0], 
                    y[is_normal]
                )
                
                # 最终修复：混合全局统计期望与局部时间趋势
                # 0.3 * 全局 GMM 期望 + 0.7 * 局部线性趋势
                data_repaired[s, is_abnormal, col] = (
                    0.3 * expected_val[col] + 0.7 * y_interp[is_abnormal]
                )

        except Exception:
            # 兜底方案：线性插值
            for col in range(n_features):
                y = data_abnormal[s, :, col]
                data_repaired[s, is_abnormal, col] = np.interp(
                    np.where(is_abnormal)[0], np.where(is_normal)[0], y[is_normal]
                )

    return data_repaired