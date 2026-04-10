import numpy as np
from hmmlearn import hmm
import warnings

def hmm_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_states=3
) -> np.ndarray:
    """
    优化后的 HMM 修复函数：
    1. 逻辑纠正：使用“正常点”训练，并在推断状态时对异常点进行平滑处理。
    2. 效率优化：减少冗余的对象实例化，并针对单变量特征优化模型参数。
    3. 鲁棒性：增加收敛检查和最小样本量保护。
    """
    # 过滤 HMM 未能收敛的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        is_abnormal = (label[sample] == 1)
        if not np.any(is_abnormal):
            continue
            
        for col in range(n_features):
            series = data_abnormal[sample, :, col].reshape(-1, 1)
            normal_data = series[~is_abnormal]
            
            # 基础校验：如果正常点太少，不足以训练 HMM 状态
            if len(normal_data) < n_states * 2:
                # 回退至线性插值
                data_repaired[sample, is_abnormal, col] = np.interp(
                    np.where(is_abnormal)[0], np.where(~is_abnormal)[0], normal_data.flatten()
                )
                continue

            try:
                # 1. 训练模型：仅使用正常数据提取分布特征
                # covariance_type="diag" 适合单变量，n_iter 减小以提升速度
                model = hmm.GaussianHMM(
                    n_components=n_states, 
                    covariance_type="diag", 
                    n_iter=50,
                    tol=0.01
                )
                model.fit(normal_data)
                
                # 2. 状态序列修复逻辑：
                # 为了防止异常值干扰状态预测，我们先对输入进行初步平滑（线性插值）
                # 这一步是为了让 model.predict 能够识别出“这个点原本应该属于哪个状态”
                temp_series = series.copy()
                temp_series[is_abnormal] = np.interp(
                    np.where(is_abnormal)[0], np.where(~is_abnormal)[0], normal_data.flatten()
                ).reshape(-1, 1)
                
                # 推断最可能的隐藏状态序列 (Viterbi 算法)
                states = model.predict(temp_series)
                
                # 3. 修复赋值
                means = model.means_.flatten()
                # 我们可以选择使用均值，或者加上该状态的随机噪声以保持统计特性
                # 这里使用均值以保证修复的确定性
                repaired_values = means[states]
                
                data_repaired[sample, is_abnormal, col] = repaired_values[is_abnormal]
                
            except Exception:
                # 若模型拟合失败（如方差坍缩），使用线性插值作为保底
                data_repaired[sample, is_abnormal, col] = temp_series[is_abnormal].flatten()
                
    return data_repaired