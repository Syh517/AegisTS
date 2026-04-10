import numpy as np
import pywt

def wavelet_denoise_repair(
    data_abnormal: np.ndarray, label: np.ndarray = None, wavelet="db4", level=None, threshold_factor=1.5
) -> np.ndarray:
    """
    自适应小波修复函数（不依赖 label）：
    1. 自动定位：通过小波高频系数的异常波动自动识别时序中的突刺和噪声。
    2. 局部识别：针对每个样本、每个特征独立计算噪声阈值。
    3. 协同修复：先初步填补 NaN，再通过小波阈值去噪获取信号基准，最后修复离群点。
    """
    data_repaired = data_abnormal.copy().astype(float)
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 预计算：小波分解允许的最大层数
    max_level = pywt.dwt_max_level(n_timestamps, pywt.Wavelet(wavelet).dec_len)
    level = min(3, max_level) if level is None else min(level, max_level)

    x_ticks = np.arange(n_timestamps)

    for s in range(n_samples):
        for c in range(n_features):
            y = data_abnormal[s, :, c]
            
            # --- Step 1: 预处理 NaN ---
            # 小波变换不支持 NaN，必须先进行初步线性插值
            is_nan = np.isnan(y)
            if np.all(is_nan): continue
            
            if np.any(is_nan):
                y_filled = np.interp(x_ticks, x_ticks[~is_nan], y[~is_nan])
            else:
                y_filled = y.copy()
            
            # --- Step 2: 小波分解 ---
            coeffs = pywt.wavedec(y_filled, wavelet=wavelet, level=level)
            
            # --- Step 3: 自动异常检测 (基于高频系数) ---
            # 最高频系数 coeffs[-1] 往往包含了噪声信息
            # 我们通过这个分布计算一个“脏数据”判定阈值
            detail_coeffs = coeffs[-1]
            # 计算稳健的偏差估计 (MAD)
            mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
            # 如果 MAD 太小（平滑信号），设为一个微小值避免失效
            mad = max(mad, 1e-6)
            
            # --- Step 4: 阈值处理与信号重构 ---
            # 使用 VisuShrink 准则进行去噪重构
            sigma = mad / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(n_timestamps))
            
            new_coeffs = [coeffs[0]] # 保护近似分量（趋势）
            for i in range(1, len(coeffs)):
                new_coeffs.append(pywt.threshold(coeffs[i], value=uthresh, mode="soft"))
            
            reconstructed = pywt.waverec(new_coeffs, wavelet)
            if len(reconstructed) != n_timestamps:
                reconstructed = reconstructed[:n_timestamps]

            # --- Step 5: 自动判定修复掩码 ---
            # 脏数据判定逻辑：
            # 1. 原始就是 NaN 的点
            # 2. 原始值与重构平滑值偏差过大的点（突刺/脏数据）
            # 这里 threshold_factor 可调，通常 1.5-3.0
            residual = np.abs(y_filled - reconstructed)
            is_dirty = (residual > (threshold_factor * uthresh)) | is_nan
            
            # --- Step 6: 局部修复 ---
            data_repaired[s, is_dirty, c] = reconstructed[is_dirty]
            
    return data_repaired