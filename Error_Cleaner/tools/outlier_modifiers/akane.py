import numpy as np
from sklearn.preprocessing import StandardScaler
from .akane0 import MultivariateCleaner, MVPatternMiner, MVRepairer
import warnings

def akane_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None, 
    window_len: int = 40, 
    k_range: tuple = (5, 15), 
    markov_order: int = 2,
    repair_method: str = 'cubic_spline',
    ppl_threshold: float = 2.0  # 自动检测阈值：PPL 超过均值的倍数
) -> np.ndarray:
    """
    自适应 Akane 修复函数（不依赖 label）：
    1. 自动挖掘：利用 MVPatternMiner 自动从混合数据中提取频繁出现的模式。
    2. 困惑度检测：利用马尔可夫模型计算每个时间步的 PPL，自动识别“不合群”的跳变点。
    3. 贪婪清洗：在 PPL 引导下，通过迭代尝试不同的修复值，使序列整体概率达到最大化。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 鲁棒标准化 ---
    scaler = StandardScaler()
    # 无 label 时，先进行初步的中位数清洗以防止模式挖掘被极端值干扰
    data_flat = data_abnormal.reshape(-1, n_features)
    medians = np.nanmedian(data_flat, axis=0)
    data_prefilled = np.where(np.isnan(data_abnormal), medians, data_abnormal)
    
    scaler.fit(data_prefilled.reshape(-1, n_features))
    all_scaled = scaler.transform(data_prefilled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    # --- 2. 建立全局模式库 (Pattern Bank) ---
    # 抽取具有代表性的片段来建立马尔可夫转移基准
    sample_size = min(n_samples, 30)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    X_ref = all_scaled[sample_indices].reshape(-1, n_features)
    
    global_miner = MVPatternMiner(window_len=window_len, stride=window_len//2)
    global_miner.auto_fit(X_ref, k_range=k_range)
    n_patterns = global_miner.n_patterns

    # --- 3. 逐样本自适应清洗 ---
    for sample in range(n_samples):
        X_scaled = all_scaled[sample].astype(float)
        
        # 定义修复算子
        repair_window = max(5, window_len // 4)
        if repair_method == 'cubic_spline':
            mv_repairer = lambda X, i: MVRepairer.cubic_spline(X, i, window=repair_window)
        else:
            mv_repairer = lambda X, i: MVRepairer.local_linear(X, i, window=repair_window)

        # --- 4. 自动识别异常点 (Candidate Selection) ---
        # 即使没有 label，我们也需要通过 PPL 初筛出“可疑”的索引，以降低计算开销
        try:
            # 初始化一个临时清洗器来获取原始 PPL 分布
            temp_cleaner = MultivariateCleaner(
                backend='markov', n_components=n_patterns, 
                markov_order=markov_order, miner=global_miner
            )
            # 计算每个点对 PPL 的贡献
            ppl_scores = temp_cleaner.calculate_ppl_profile(X_scaled)
            
            # 自动判定：超过平均 PPL 一定倍数的点，或者原始数据为 NaN 的点
            mask_nan = np.isnan(data_abnormal[sample]).any(axis=1)
            candidate_indices = np.where((ppl_scores > np.mean(ppl_scores) * ppl_threshold) | mask_nan)[0]
            
            if len(candidate_indices) == 0:
                continue

            # --- 5. 执行贪婪清洗 ---
            cleaner = MultivariateCleaner(
                backend='markov', 
                n_components=n_patterns, 
                markov_order=markov_order,
                repairer=mv_repairer,
                miner=global_miner,
                candidate_indices=candidate_indices
            )
            
            # 设置 budget。在无 label 时，我们允许清洗所有可疑点
            X_cleaned_scaled, _, _ = cleaner.greedy_clean(X_scaled, budget=len(candidate_indices))
            
            # --- 6. 写回 ---
            data_repaired[sample] = scaler.inverse_transform(X_cleaned_scaled)
            
        except Exception as e:
            # 如果某样本计算失败（通常由于序列过短），则使用简单的线性插值保底
            is_nan = np.isnan(data_abnormal[sample]).any(axis=1)
            if np.any(is_nan):
                for c in range(n_features):
                    y = data_abnormal[sample, :, c]
                    data_repaired[sample, is_nan, c] = np.interp(
                        np.where(is_nan)[0], np.where(~is_nan)[0], y[~is_nan]
                    )
            continue
            
    return data_repaired