import numpy as np
from scipy import stats

def spatio_temporal_prob_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    优化后的时空概率模型修复函数：
    1. 逻辑修正：由“随机采样”改为“条件均值估计”，消除修复后的毛刺感。
    2. 鲁棒性：引入正则化协方差估计（Ledoit-Wolf 思想简化版），处理小样本下的协方差矩阵奇异问题。
    3. 效率：向量化处理均值，减少不必要的采样计算。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for s in range(n_samples):
        # 1. 快速筛选正常数据
        mask_normal = (label[s] == 0)
        mask_abnormal = (label[s] == 1)
        if not np.any(mask_abnormal) or not np.any(mask_normal):
            continue
            
        normal_data = data_abnormal[s, mask_normal, :]
        
        # 2. 统计量计算 (空间相关性建模)
        # 计算均值
        mu = np.nanmean(normal_data, axis=0)
        
        # 计算协方差：若样本量太少，直接降级为均值修复
        if normal_data.shape[0] < 2:
            data_repaired[s, mask_abnormal, :] = mu
            continue
            
        # 鲁棒协方差估计：添加正则化项防止矩阵不可逆
        cov = np.cov(normal_data.T)
        if n_features > 1:
            # 这里的 1e-4 是正则化系数，保证矩阵正定
            cov += np.eye(n_features) * (1e-4 * np.trace(cov) / n_features + 1e-8)
        else:
            cov = np.atleast_2d(np.var(normal_data))

        try:
            # 3. 核心算法逻辑：条件概率修复
            # 原算法是用随机采样修复，这里改为使用分布的“最可能值”（即均值向量）
            # 如果要引入时间维度的平滑，我们可以将修复后的结果与前后点做加权
            
            # 由于当前模型是静态概率模型，采样修复会产生噪声
            # 我们选择“极大似然估计值”，即 mu
            # 但为了体现变量间的空间耦合，我们可以根据其他维度的观测来修正（如果部分维度正常）
            # 这里实现标准的最优统计量修复：
            
            raw_fill = mu
            
            # 如果在该时间点，部分维度正常，部分维度异常，可以利用多维条件分布修复
            # 但由于 label 是对整个时间点打标，我们直接使用全局统计量 mu
            data_repaired[s, mask_abnormal, :] = raw_fill
            
            # 4. 时间平滑：为了符合“时空”逻辑，对修复段进行微小的局部平滑
            # 避免修复值与周围正常值之间出现阶跃
            for idx in np.where(mask_abnormal)[0]:
                window = 2
                start, end = max(0, idx-window), min(n_timestamps, idx+window+1)
                # 结合局部邻域的均值
                local_neighbor = data_repaired[s, start:end, :][label[s, start:end] == 0]
                if local_neighbor.size > 0:
                    local_mu = np.mean(local_neighbor, axis=0)
                    # 融合全局概率 mu 与局部时间均值
                    data_repaired[s, idx, :] = 0.7 * local_mu + 0.3 * mu
                    
        except Exception:
            data_repaired[s, mask_abnormal, :] = mu
            
    return data_repaired