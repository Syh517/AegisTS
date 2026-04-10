import numpy as np
from scipy import interpolate

def abnormal_sequence_interpolation_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, method="linear", window_size=10, threshold=3
) -> np.ndarray:
    """
    自适应序列插值修复函数：
    1. 自动检测：集成 Hampel Filter 逻辑，利用中位数偏差自动识别“突刺”脏数据。
    2. 统一掩码：自动识别并合并 NaN, Inf 以及统计学离群点。
    3. 高效插值：保持原有的全局掩码插值逻辑，兼顾边界外推。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    x_full = np.arange(n_timestamps)
    
    for s in range(n_samples):
        # --- Step 1: 自动异常检测 (Hampel Filter 思想) ---
        # 为每个特征独立计算异常掩码
        sample_abnormal_mask = np.zeros((n_timestamps, n_features), dtype=bool)
        
        for c in range(n_features):
            series = data_abnormal[s, :, c]
            
            # a. 识别基础无效值 (NaN/Inf)
            invalid_mask = ~np.isfinite(series)
            
            # b. 识别统计异常值 (脏数据)
            # 使用滑动窗口中位数过滤，计算与中位数的偏差
            # 这种方法对时序中的突发噪声极其有效
            try:
                # 简单的移动中位数实现
                padded = np.pad(series, (window_size, window_size), mode='edge')
                # 构造滑动窗口矩阵 (向量化操作)
                windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * window_size + 1)
                medians = np.nanmedian(windows, axis=1)[:n_timestamps]
                mads = np.nanmedian(np.abs(windows - medians[:, None]), axis=1)[:n_timestamps]
                
                # 阈值判定：偏离中位数超过 threshold * MAD 的视为脏数据
                # 默认 threshold=3 约等于正态分布下的 3-sigma
                stats_outlier = np.abs(series - medians) > (threshold * mads)
            except:
                stats_outlier = np.zeros(n_timestamps, dtype=bool)
                
            sample_abnormal_mask[:, c] = invalid_mask | stats_outlier

        # --- Step 2: 遍历特征进行修复 ---
        for c in range(n_features):
            is_feat_abnormal = sample_abnormal_mask[:, c]
            if not np.any(is_feat_abnormal):
                continue
                
            is_feat_normal = ~is_feat_abnormal
            x_normal = x_full[is_feat_normal]
            y_full = data_abnormal[s, :, c]
            y_normal = y_full[is_feat_normal]
            
            # 如果该特征没有可用的正常点，尝试用均值或跳过
            if x_normal.size < 2:
                if x_normal.size == 1:
                    data_repaired[s, is_feat_abnormal, c] = y_normal[0]
                continue

            # --- Step 3: 执行插值 ---
            if method == "linear":
                # np.interp 处理速度最快，自带边界填充
                data_repaired[s, is_feat_abnormal, c] = np.interp(
                    x_full[is_feat_abnormal], 
                    x_normal, 
                    y_normal
                )
            else:
                # 针对 spline, cubic 等方法
                try:
                    f = interpolate.interp1d(
                        x_normal, 
                        y_normal, 
                        kind=method, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                    )
                    data_repaired[s, is_feat_abnormal, c] = f(x_full[is_feat_abnormal])
                except ValueError:
                    # 如果点数不足以支持复杂的 kind (如 cubic 需要至少4个点)，回退到线性
                    data_repaired[s, is_feat_abnormal, c] = np.interp(
                        x_full[is_feat_abnormal], x_normal, y_normal
                    )
                
    return data_repaired