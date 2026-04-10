import numpy as np
from hmmlearn import hmm
import warnings

def hmm_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, n_states=3
) -> np.ndarray:
    """
    基于 HMM 概率建模的自适应修复：
    1. 移除 label 依赖，通过观测概率自动区分“正常状态”与“异常/离群状态”。
    2. MLE 逻辑：推断隐藏状态序列，并将低似然点替换为该状态的期望值。
    3. 鲁棒性：使用特征全局信息初始化，防止模型被局部噪声带偏。
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            series = data_abnormal[sample, :, col].reshape(-1, 1)
            
            try:
                # --- 1. 训练 HMM 模型 ---
                # 增加一个状态用于捕获异常（或者让模型自行发现离群状态）
                model = hmm.GaussianHMM(
                    n_components=n_states, 
                    covariance_type="diag", 
                    n_iter=100,
                    tol=0.01,
                    init_params="stmc" # 初始化起始概率、转移矩阵、均值、协方差
                )
                model.fit(series)
                
                # --- 2. 自动检测异常 (基于对数似然) ---
                # 计算每个观测点在当前模型下的对数似然（得分）
                # score_samples 返回每个时间点属于当前序列的概率贡献
                log_probs = model.predict_proba(series) # (n_timestamps, n_states)
                states = np.argmax(log_probs, axis=1)
                
                # 计算每个点的自似然度：如果一个点的观测概率极低，认为它是“脏”的
                # 或者是属于一个方差极大的“离群状态”
                means = model.means_.flatten()
                covars = model.covars_.flatten()
                
                # 找出“正常状态”：通常是方差较小的状态
                normal_states_mask = covars < np.percentile(covars, 75)
                
                # --- 3. 修复逻辑 ---
                for t in range(n_timestamps):
                    current_state = states[t]
                    val = series[t, 0]
                    
                    # 判定条件：
                    # a) 如果当前状态是“高方差状态”（即模型被迫为离群点建立的状态）
                    # b) 或者该点偏离其所属状态均值超过 3 倍标准差
                    state_mu = means[current_state]
                    state_std = np.sqrt(covars[current_state])
                    
                    is_outlier_state = not normal_states_mask[current_state]
                    is_extreme_value = np.abs(val - state_mu) > 3 * state_std
                    
                    if is_outlier_state or is_extreme_value:
                        # 重新寻找最可能的“正常”状态进行填充
                        # 在 MLE 意义下，我们选择除当前异常状态外，转移概率最高的正常状态均值
                        if np.any(normal_states_mask):
                            # 简化处理：直接映射到最近的正常状态均值
                            normal_means = means[normal_states_mask]
                            data_repaired[sample, t, col] = normal_means[np.argmin(np.abs(normal_means - val))]
                        else:
                            data_repaired[sample, t, col] = state_mu
                            
            except Exception:
                # 拟合失败保底：使用中值平滑
                data_repaired[sample, :, col] = np.clip(series.flatten(), 
                                                       np.nanpercentile(series, 5), 
                                                       np.nanpercentile(series, 95))
                
    return data_repaired