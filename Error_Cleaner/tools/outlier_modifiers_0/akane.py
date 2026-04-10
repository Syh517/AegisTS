import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .akane0 import MultivariateCleaner, MVPatternMiner, MVRepairer
import warnings

def akane_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window_len: int = 40, 
    k_range: tuple = (5, 15), 
    markov_order: int = 2,
    repair_method: str = 'cubic_spline'
) -> np.ndarray:
    """
    优化后的 Akane 困惑度指导清洗：
    1. 引入全局标准化。
    2. 全局模式发现加速：若样本相似，仅在全局采样点上拟合一次模式。
    3. 修复逻辑强化：确保 label=1 的点被强制重构。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 全局标准化 (提升聚类模式识别的准确度) ---
    scaler = StandardScaler()
    # 展平并拟合正常点
    normal_points = data_abnormal[label == 0]
    if len(normal_points) == 0:
        scaler.fit(data_abnormal.reshape(-1, n_features))
    else:
        scaler.fit(normal_points)
    
    # --- 2. 预提取全局模式 (效率优化点) ---
    # 如果 n_samples 很大，不再逐一 auto_fit，而是从全局正常数据中抽样拟合一个通用的 Miner
    print("[akane_repair] Pre-fitting global pattern miner...")
    all_scaled = scaler.transform(data_abnormal.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    # 抽取部分样本来建立全局模式，避免 OOM
    sample_indices = np.random.choice(n_samples, min(n_samples, 20), replace=False)
    X_ref = all_scaled[sample_indices].reshape(-1, n_features)
    
    global_miner = MVPatternMiner(window_len=window_len, stride=window_len//2)
    global_miner.auto_fit(X_ref, k_range=k_range)
    n_patterns = global_miner.n_patterns

    # --- 3. 逐样本处理 ---
    for sample in range(n_samples):
        mask_anomaly = label[sample] == 1
        candidate_indices = np.where(mask_anomaly)[0]
        
        if len(candidate_indices) == 0:
            continue

        X_scaled = all_scaled[sample]

        # --- 4. 修复算子定义 ---
        # 优化点：window 大小应随 window_len 动态调整
        repair_window = max(5, window_len // 4)
        if repair_method == 'cubic_spline':
            mv_repairer = lambda X, i: MVRepairer.cubic_spline(X, i, window=repair_window)
        else:
            mv_repairer = lambda X, i: MVRepairer.local_linear(X, i, window=repair_window)

        # --- 5. 执行清洗 ---
        # 使用预训练的 global_miner 提高效率
        cleaner = MultivariateCleaner(
            backend='markov', 
            n_components=n_patterns, 
            markov_order=markov_order,
            repairer=mv_repairer,
            miner=global_miner,
            candidate_indices=candidate_indices
        )
        
        # greedy_clean 内部会根据 PPL 下降程度执行修复
        # 强制 budget=len(candidate_indices) 确保所有标签点都被尝试修复
        try:
            X_cleaned_scaled, _, _ = cleaner.greedy_clean(X_scaled, budget=len(candidate_indices))
            
            # --- 6. 逆变换并写回 ---
            data_repaired[sample] = scaler.inverse_transform(X_cleaned_scaled)
        except Exception as e:
            warnings.warn(f"Akane repair failed at sample {sample}: {e}")
            # 失败则保留原值或简单的线性插值作为保底
            continue
            
    return data_repaired