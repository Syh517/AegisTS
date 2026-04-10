import numpy as np
from sklearn.mixture import GaussianMixture
import warnings

def gaussian_mixture_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, n_components=5
) -> np.ndarray:
    """
    基于 GMM 似然判别的自适应修复函数：
    1. 移除 label 依赖：利用 score_samples 计算每个样本点的对数似然度，自动识别离群点。
    2. MLE 修复逻辑：通过计算后验概率 (responsibilities)，将异常点替换为各高斯分量的加权期望。
    3. 动态平滑：结合时间轴的局部中值，防止修复后的数据产生阶跃，同时保持多模态特性。
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    data_repaired = data_abnormal.copy().astype(float)
    
    # 展平以便全局学习多模态分布特征
    flat_data = data_abnormal.reshape(-1, n_features)

    try:
        # --- 1. 拟合 GMM 模型 (参数估计) ---
        # 即使数据中有噪声，GMM 也能通过 EM 算法找到主要的正常聚类
        gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type="diag", 
            max_iter=100, 
            random_state=42
        )
        gmm.fit(flat_data)
        
        # --- 2. 自动异常识别 (似然度判定) ---
        # 计算每个点在当前分布下的对数似然得分
        log_likelihoods = gmm.score_samples(flat_data)
        
        # 设定阈值：通常对数似然最低的 5% 被视为异常噪声
        # 也可以使用：mean(scores) - 3 * std(scores) 作为动态阈值
        threshold = np.percentile(log_likelihoods, 5)
        is_dirty_flat = log_likelihoods < threshold
        
        # --- 3. 基于极大似然的修复 (推断) ---
        # 计算所有点属于各个组件的后验概率 (n_points, n_components)
        # 这决定了修复时应该“倾向于”哪个高斯中心
        responsibilities = gmm.predict_proba(flat_data)
        
        # 预计算全局加权期望作为基础参考
        # 形状: (n_points, n_features)
        expected_values = responsibilities @ gmm.means_
        
        # 将展平的脏点标志还原为 (n_samples, n_timestamps)
        is_dirty = is_dirty_flat.reshape(n_samples, n_timestamps)

        for s in range(n_samples):
            for t in range(n_timestamps):
                if not is_dirty[s, t]:
                    continue
                
                # 获取该点的 MLE 修复基础值
                idx_flat = s * n_timestamps + t
                mle_val = expected_values[idx_flat]
                
                # --- 时间约束平滑 ---
                # 寻找该点周围的局部时间上下文，以防修复值与前后脱节
                w = 3
                t_start, t_end = max(0, t-w), min(n_timestamps, t+w+1)
                # 仅参考周围被判定为“正常”的点
                local_normal = data_abnormal[s, t_start:t_end, :][~is_dirty[s, t_start:t_end]]
                
                if local_normal.size > 0:
                    local_trend = np.median(local_normal, axis=0)
                    # 融合：40% 全局分布期望 + 60% 局部时间趋势
                    data_repaired[s, t, :] = 0.4 * mle_val + 0.6 * local_trend
                else:
                    data_repaired[s, t, :] = mle_val

    except Exception:
        # 拟合失败保底：滑动中值滤波
        import scipy.ndimage as ndimage
        for c in range(n_features):
            data_repaired[:, :, c] = ndimage.median_filter(data_abnormal[:, :, c], size=(1, 5))

    return data_repaired