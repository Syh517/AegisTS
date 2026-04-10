# ======================================================
# ========    基于平滑的方法 ============================
# ======================================================


import pandas as pd
import numpy as np
from scipy import interpolate
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR
from shapely.geometry import LineString


# ------------------------------ 1. Moving Average ------------------------------
def moving_average_repair(
    data_abnormal: np.ndarray, label: np.ndarray, window=3
) -> np.ndarray:
    """
    使用移动平均方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: 移动平均窗口大小
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0  # 正常点
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            # 将异常点设为 NaN，避免参与平均
            series[~mask] = np.nan
            # 使用滚动平均，自动忽略 NaN
            series_pd = pd.Series(series)
            smoothed = series_pd.rolling(window=window, center=True, min_periods=1).mean()
            data_repaired[sample, label[sample] == 1, col] = smoothed.values[label[sample] == 1]
        
    return data_repaired


# ------------------------------ 2. AutoRegressive (AR) ------------------------------
def autoregressive_repair(
    data_abnormal: np.ndarray, label: np.ndarray, lags=3
) -> np.ndarray:
    """
    使用自回归模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        lags: 自回归模型的滞后阶数
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            # 剔除异常点（设为 NaN）
            series[~mask] = np.nan
            # 插值或前向填充以平稳序列（可选）
            series_pd = pd.Series(series)
            series = series_pd.interpolate().fillna(method="bfill").fillna(method="ffill").values

            model = AutoReg(series, lags=lags, trend="n", old_names=False)
            fitted = model.fit()
            preds = fitted.predict(start=0, end=len(series) - 1)
            data_repaired[sample, label[sample] == 1, col] = preds[label[sample] == 1]
        
    return data_repaired


# ------------------------------ 3. ARMA ------------------------------
def arma_repair(
    data_abnormal: np.ndarray, label: np.ndarray, p=2, q=1
) -> np.ndarray:
    """
    使用ARMA模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        p: AR部分的阶数
        q: MA部分的阶数
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            # 剔除异常点
            series[~mask] = np.nan
            # 插值补全用于建模
            series_pd = pd.Series(series)
            series_interp = (
                series_pd.interpolate().fillna(method="bfill").fillna(method="ffill")
            ).values

            model = ARIMA(series_interp, order=(p, 0, q))
            fitted = model.fit()
            preds = fitted.fittedvalues  # 或 predict()
            data_repaired[sample, label[sample] == 1, col] = preds[label[sample] == 1]
        
    return data_repaired


# ------------------------------ 4. Kalman Filter ------------------------------
def kalman_filter_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    使用卡尔曼滤波修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()

    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0  # 正常点
        abnormal_mask = label[sample] == 1  # 异常点待修复
        for col in range(n_features):
            series = data_abnormal[sample, :, col]
            measurements = series.astype(float).copy()  # 确保浮点型
            measurements[~mask] = np.nan  # 将异常点标记为缺失

            # 检查是否有有效值用于初始化
            valid_vals = measurements[mask]
            if len(valid_vals) == 0 or np.all(np.isnan(valid_vals)):
                # 若无正常点，跳过修复（保持原值）
                continue

            initial_mean = np.nanmean(valid_vals)
            initial_cov = np.nanvar(valid_vals)

            # 处理全 nan 列的情况
            if np.isnan(initial_mean) or np.isinf(initial_mean):
                initial_mean = 0.0
            if np.isnan(initial_cov) or np.isinf(initial_cov) or initial_cov <= 1e-8:
                initial_cov = 1.0

            # 构建 Kalman Filter
            kf = KalmanFilter(
                initial_state_mean=initial_mean,
                initial_state_covariance=initial_cov,
                transition_matrices=[1.0],  # 恒定模型
                observation_matrices=[1.0],
                observation_covariance=1.0,  # R
                transition_covariance=1e-3,  # Q，过程噪声小
            )

            try:
                # ⚠️ 关键：EM 不接受 NaN！但我们传入含 NaN 的 measurements？
                # 实际上 pykalman 支持，但需确保没有 inf
                measurements = np.where(np.isfinite(measurements), measurements, np.nan)

                # 使用 EM 学习参数（允许 NaN）
                learned_kf = kf.em(measurements, n_iter=10)
                smoothed_state_means, _ = learned_kf.smooth(measurements)

                # 填充异常点
                repair_values = smoothed_state_means.flatten()[abnormal_mask]
                data_repaired[sample, abnormal_mask, col] = repair_values

            except Exception as e:
                print(f"Kalman filtering failed on sample {sample}, column '{col}': {str(e)}")
                # 失败时可用线性插值等备用方法
                temp_series = pd.Series(measurements)
                temp_series[abnormal_mask] = np.nan
                temp_series = temp_series.interpolate(
                    method="linear", limit_direction="both"
                )
                data_repaired[sample, abnormal_mask, col] = temp_series.values[abnormal_mask]
            
    return data_repaired


# ------------------------------ 5. Interpolation ------------------------------
def interpolation_repair(
    data_abnormal: np.ndarray, label: np.ndarray, method="linear"
) -> np.ndarray:
    """
    使用插值方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        method: 插值方法
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            x = np.arange(len(series))
            mask = label[sample] == 0
            f = interpolate.interp1d(
                x[mask], series[mask], kind=method, fill_value="extrapolate"
            )
            data_repaired[sample, label[sample] == 1, col] = f(x[label[sample] == 1])
        
    return data_repaired


# ------------------------------ 6. State-space Model (Kalman variant) ------------------------------
def state_space_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    使用状态空间模型（卡尔曼滤波变体）修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            series[~mask] = np.nan  # 设为缺失
            series_pd = pd.Series(series)
            series_interp = (
                series_pd.interpolate().fillna(method="bfill").fillna(method="ffill")
            ).values

            initial_mean = series[mask][0] if mask.any() else series_interp[0]

            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=initial_mean,
                observation_covariance=1,
                transition_covariance=0.01,
            )
            state_means, _ = kf.em(series_interp, n_iter=10).smooth(
                series_interp
            )
            data_repaired[sample, label[sample] == 1, col] = state_means[label[sample] == 1].flatten()
        
    return data_repaired


# ------------------------------ 7. Trajectory Simplification (Douglas-Peucker) ------------------------------
def trajectory_simplification_repair(
    data_abnormal: np.ndarray, label: np.ndarray, tolerance=0.01
) -> np.ndarray:
    """
    使用轨迹简化方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        tolerance: 简化容差
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        x_coords = np.arange(n_timestamps).astype(float)
        mask = label[sample] == 0  # 仅使用正常点构建轨迹
        for col in range(n_features):
            y_vals = data_abnormal[sample, :, col].astype(float)

            # 只用正常点构建 LineString
            normal_x = x_coords[mask]
            normal_y = y_vals[mask]
            if len(normal_x) < 2:
                # 太少正常点，无法简化
                continue
            points = [(normal_x[i], normal_y[i]) for i in range(len(normal_x))]
            line = LineString(points)

            simplified = line.simplify(tolerance, preserve_topology=False)
            x_s, y_s = np.array(simplified.xy)

            # 排序
            sorted_idx = np.argsort(x_s)
            x_s_sorted = x_s[sorted_idx]
            y_s_sorted = y_s[sorted_idx]

            try:
                f = interpolate.interp1d(
                    x_s_sorted,
                    y_s_sorted,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            except ValueError:
                if len(x_s_sorted) == 1:
                    data_repaired[sample, label[sample] == 1, col] = y_s_sorted[0]
                    continue
                else:
                    raise

            anomaly_x = x_coords[label[sample] == 1]
            y_pred = f(anomaly_x)
            y_pred = np.nan_to_num(y_pred, nan=y_s_sorted.mean())

            data_repaired[sample, label[sample] == 1, col] = y_pred
        
    return data_repaired


# ------------------------------ 8. Exponential Smoothing ------------------------------
def exponential_smoothing_repair(
    data_abnormal: np.ndarray, label: np.ndarray, alpha=0.3
) -> np.ndarray:
    """
    使用指数平滑方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        alpha: 平滑参数
        
    返回:
        修复后的三维numpy数组
    """
    # 处理所有样本
    data_repaired = data_abnormal.copy()
    
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask = label[sample] == 0
        for col in range(n_features):
            series = data_abnormal[sample, :, col].copy()
            # 剔除异常点
            series[~mask] = np.nan
            # 插值补全用于拟合
            series_pd = pd.Series(series)
            series_filled = (
                series_pd.interpolate().fillna(method="bfill").fillna(method="ffill")
            ).values

            model = ExponentialSmoothing(series_filled, trend=None, seasonal=None)
            fitted = model.fit(smoothing_level=alpha, optimized=False)
            preds = fitted.fittedvalues
            data_repaired[sample, label[sample] == 1, col] = preds[label[sample] == 1]
        
    return data_repaired


# ------------------------------ 9. Support Vector Regression ------------------------------
def svr_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    使用支持向量回归 (SVR) 修复异常值
    
    参数:
        data_abnormal: 三维 numpy 数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维 numpy 数组，形状为 (n_samples, n_timestamps)，1=异常，0=正常
        
    返回:
        修复后的三维 numpy 数组，形状同 data_abnormal
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    # 时间索引作为 x（对所有样本共享）
    x = np.arange(n_timestamps).reshape(-1, 1)

    for sample in range(n_samples):
        for col in range(n_features):
            y = data_abnormal[sample, :, col]
            mask_normal = (label[sample] == 0)  # 正常点掩码

            # 如果正常点太少，跳过或用简单插值（可选）
            if np.sum(mask_normal) < 2:
                # 退化处理：用前后插值
                series_pd = pd.Series(y)
                series_filled = (
                    series_pd.interpolate()
                    .fillna(method="bfill")
                    .fillna(method="ffill")
                ).values
                data_repaired[sample, :, col] = series_filled
                continue

            # 用正常点训练 SVR
            svr = SVR(kernel="rbf", C=10, epsilon=0.1)
            svr.fit(x[mask_normal], y[mask_normal])
            
            # 预测所有时间点
            y_pred = svr.predict(x)
            
            # 仅将预测值填入异常位置
            mask_abnormal = (label[sample] == 1)
            data_repaired[sample, mask_abnormal, col] = y_pred[mask_abnormal]

    return data_repaired



# ======================================================
# ========    基于统计的方法 ============================
# ======================================================


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from hmmlearn import hmm
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer


# ------------------------------ 1. Maximum Likelihood Estimation (MLE) ------------------------------
def maximum_likelihood_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    使用最大似然估计修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            normal_values = data_abnormal[sample, label[sample] == 0, col]
            mu, sigma = np.mean(normal_values), np.std(normal_values)
            mle_values = np.random.normal(mu, sigma, n_timestamps)
            data_repaired[sample, label[sample] == 1, col] = mle_values[label[sample] == 1]
    return data_repaired


# ------------------------------ 2. Bayesian Model (Bayesian Ridge) ------------------------------
def bayesian_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    使用贝叶斯模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    x = np.arange(n_timestamps).reshape(-1, 1)
    
    for sample in range(n_samples):
        for col in range(n_features):
            y = data_abnormal[sample, :, col]
            mask = label[sample] == 0
            model = BayesianRidge()
            model.fit(x[mask], y[mask])
            y_pred = model.predict(x)
            data_repaired[sample, label[sample] == 1, col] = y_pred[label[sample] == 1]
    return data_repaired


# ------------------------------ 3. Markov Model (First-order transition estimation) ------------------------------
def markov_model_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    使用线性插值 + 马尔可夫模型修正 进行异常修复
    优势：
        - 线性插值：保证连续异常段平滑过渡
        - 马尔可夫修正：调整插值结果，使其更符合正常变化趋势（避免过度平滑）
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        n = n_timestamps
        normal_mask = label[sample] == 0
        abnormal_mask = label[sample] == 1

        for col in range(n_features):
            original = data_abnormal[sample, :, col]
            repaired = original.copy()

            # Step 1: 提取正常点用于建模
            normal_indices = np.where(normal_mask)[0]
            if len(normal_indices) == 0:
                continue  # 无正常点，无法修复

            # Step 2: 拟合马尔可夫转移模型（只用正常点之间的差分）
            valid_diffs = []
            for i in range(n - 1):
                if normal_mask[i] and normal_mask[i + 1]:
                    valid_diffs.append(original[i + 1] - original[i])

            if len(valid_diffs) == 0:
                # 无有效转移，退化为线性插值
                repaired[abnormal_mask] = np.interp(
                    np.where(abnormal_mask)[0], normal_indices, original[normal_indices]
                )
            else:
                mu = np.mean(valid_diffs)  # 平均变化量
                sigma = max(np.std(valid_diffs), 1e-6)

                # Step 3: 先用线性插值获得平滑初值（处理连续异常）
                # 将异常点设为 NaN，便于插值
                temp_series = repaired.astype(float).copy()
                temp_series[abnormal_mask] = np.nan
                # 线性插值（前后向）
                temp_series_pd = pd.Series(temp_series)
                temp_series = (
                    temp_series_pd
                    .interpolate(method="linear", limit_direction="both")
                    .values
                )

                # Step 4: 马尔可夫修正 —— 调整插值点，使其更符合正常变化趋势
                for idx in np.where(abnormal_mask)[0]:
                    if idx == 0:
                        repaired[0] = temp_series[0]  # 首点直接用插值
                    else:
                        prev_repaired = repaired[idx - 1]  # 已修复的前一个值
                        # 预测当前值：X_t = X_{t-1} + Δμ
                        predicted = prev_repaired + mu
                        # 插值提供平滑，马尔可夫提供趋势 —— 可加权融合
                        alpha = 0.5  # 权重：0=纯插值，1=纯马尔可夫
                        repaired[idx] = alpha * predicted + (1 - alpha) * temp_series[idx]

                # 可选：对修复后序列做简单平滑（如移动平均）防止震荡
                # repaired = pd.Series(repaired).rolling(3, center=True, min_periods=1).mean().values

            data_repaired[sample, :, col] = repaired

    return data_repaired


# ------------------------------ 4. Hidden Markov Model (Gaussian emissions) ------------------------------
def hmm_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_states=3
) -> np.ndarray:
    """
    使用隐马尔可夫模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_states: 隐藏状态数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            x = data_abnormal[sample, :, col].reshape(-1, 1)
            model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="diag", n_iter=100
            )
            model.fit(x[label[sample] == 0])
            repaired = x.copy()
            means = model.means_.flatten()
            states = model.predict(x)
            for i in np.where(label[sample] == 1)[0]:
                repaired[i] = means[states[i] % len(means)]
            data_repaired[sample, :, col] = repaired.flatten()
    return data_repaired


# ------------------------------ 5. SMURF (简单近似：基于时空邻域均值替代) ------------------------------
def smurf_repair(
    data_abnormal: np.ndarray, label: np.ndarray, window=3
) -> np.ndarray:
    """
    使用SMURF方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: 窗口大小
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        for col in range(n_features):
            for i in np.where(label[sample] == 1)[0]:
                start, end = max(0, i - window), min(n_timestamps, i + window + 1)
                neighbors = data_abnormal[sample, start:end, col][label[sample, start:end] == 0]
                if len(neighbors) > 0:
                    data_repaired[sample, i, col] = np.mean(neighbors)
    return data_repaired


# ------------------------------ 6. Spatio-Temporal Probabilistic Model (简单多变量正态建模) ------------------------------
def spatio_temporal_prob_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    使用时空概率模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        normal_data = data_abnormal[sample, label[sample] == 0]
        mu = np.mean(normal_data, axis=0)
        cov = np.cov(normal_data.T)
        mvn = multivariate_normal(mean=mu, cov=cov)
        repaired_values = mvn.rvs(size=n_timestamps)
        data_repaired[sample, label[sample] == 1, :] = repaired_values[label[sample] == 1]
    return data_repaired


# ------------------------------ 7. Expectation-Maximization (EM-GMM) ------------------------------
def em_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_components=2
) -> np.ndarray:
    """
    使用期望最大化算法（GMM）修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_components: GMM组件数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        normal_data = data_abnormal[sample, label[sample] == 0]
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", max_iter=200
        )
        gmm.fit(normal_data)
        repaired = gmm.sample(n_timestamps)[0]
        data_repaired[sample, label[sample] == 1, :] = repaired[label[sample] == 1]
    return data_repaired


# ------------------------------ 8. Relationship-dependent Network (KMeans-based approximation) ------------------------------
def relationship_network_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_clusters=3
):
    """
    使用关系依赖网络（基于KMeans）修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_clusters: 聚类数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = label[sample] == 0

        if not mask_normal.any():
            continue

        # 初始化插补器：仅用正常样本训练
        imputer = KNNImputer(n_neighbors=min(5, mask_normal.sum()))
        imputer.fit(data_abnormal[sample, mask_normal, :])  # 只在正常样本上 fit

        # 对整个数据集进行填补
        data_filled = imputer.transform(data_abnormal[sample])

        # 在填补后的正常样本上聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(
            data_filled[mask_normal]
        )
        centers = kmeans.cluster_centers_
        assigned = kmeans.predict(data_filled)

        # 修复异常点
        for i in np.where(label[sample] == 1)[0]:
            cluster = assigned[i]
            data_repaired[sample, i, :] = centers[cluster]

    return data_repaired


# ------------------------------ 9. Gaussian Mixture Model ------------------------------
def gaussian_mixture_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_components=2
) -> np.ndarray:
    """
    使用高斯混合模型修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_components: GMM组件数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        normal_data = data_abnormal[sample, label[sample] == 0]
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="full", max_iter=100
        )
        gmm.fit(normal_data)
        repaired_samples, _ = gmm.sample(n_timestamps)
        data_repaired[sample, label[sample] == 1, :] = repaired_samples[label[sample] == 1]
    return data_repaired


# ------------------------------ 10. Iterative Minimum Repairing ------------------------------
from tqdm import tqdm

# def imr_repair(
#     data_abnormal: np.ndarray, label: np.ndarray, p=3, delta=1e-5, max_iter=50
# ) -> np.ndarray:
#     """
#     使用迭代最小修复方法修复异常值
    
#     参数:
#         data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
#         label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
#         p: AR模型阶数
#         delta: 收敛阈值
#         max_iter: 最大迭代次数
        
#     返回:
#         修复后的三维numpy数组
#     """

#     data_repaired = data_abnormal.copy()
#     n_samples, n_timestamps, n_features = data_abnormal.shape
    
#     # 只处理第一个特征（单变量）
#     if n_features > 1:
#         print("警告：IMR方法只适用于单变量时间序列，将只处理第一个特征")
    
#     for sample in range(n_samples):
#         y_dirty = data_abnormal[sample, :, 0].astype(float)
#         label_bool = (label[sample] == 1)  # True for abnormal points
#         n = len(y_dirty)

#         if np.sum(~label_bool) < p + 1:
#             # 正常点太少，无法拟合AR(p)模型
#             print(
#                 f"Warning: Not enough normal points for AR({p}) model in sample {sample}. Skipping."
#             )
#             continue

#         # 使用正常点拟合AR(p)模型
#         normal_mask = ~label_bool
#         normal_values = y_dirty[normal_mask]

#         # 为了拟合AR模型，需要创建滞后变量
#         # 例如对于 AR(1): y_t = c + phi1 * y_{t-1} + error
#         # 对于 AR(p): y_t = c + phi1 * y_{t-1} + ... + phi_p * y_{t-p} + error
#         # 我们需要找到所有连续的正常点序列来构建训练数据

#         # 简化方法：对正常点序列进行插值以填补缺失的滞后值
#         temp_series = pd.Series(y_dirty)
#         temp_series[label_bool] = np.nan
#         temp_series_interp = (
#             temp_series.interpolate(method="linear")
#             .fillna(method="bfill")
#             .fillna(method="ffill")
#         )
#         y_interp = temp_series_interp.values

#         X_ar = np.zeros((n - p, p))
#         y_ar = np.zeros((n - p,))

#         for i in range(p, n):
#             X_ar[i - p] = y_interp[i - p : i]  # Use interpolated values for lags
#             y_ar[i - p] = y_interp[i]

#         # Only train on rows where the target (y_ar) is a normal point
#         train_mask = normal_mask[p:n]  # Mask for y_ar corresponding to normal points
#         if train_mask.sum() >= p + 1:  # Need enough training samples
#             X_train = X_ar[train_mask]
#             y_train = y_ar[train_mask]

#             try:
#                 from sklearn.linear_model import LinearRegression

#                 ar_model = LinearRegression()
#                 ar_model.fit(X_train, y_train)

#                 # Predict for all positions (including abnormal points)
#                 y_pred = np.full(n, np.nan)
#                 y_pred[p:] = ar_model.predict(X_ar)

#                 # For abnormal points, use the AR prediction
#                 y_clean = y_dirty.copy()
#                 abn_mask = label_bool
#                 y_clean[abn_mask] = y_pred[abn_mask]

#                 # If AR prediction is NaN (e.g., due to initial lags), fall back to interpolation
#                 nan_mask = np.isnan(y_clean) & abn_mask
#                 if nan_mask.any():
#                     y_clean[nan_mask] = temp_series_interp[nan_mask]

#             except Exception as e:
#                 print(
#                     f"AR model fitting failed for sample {sample}: {e}. Using interpolation."
#                 )
#                 y_clean = temp_series_interp
#         else:
#             # If not enough normal points to train AR, fall back to interpolation
#             print(
#                 f"Not enough normal points after lagging for AR model in sample {sample}. Using interpolation."
#             )
#             y_clean = temp_series_interp

#         data_repaired[sample, :, 0] = y_clean

#     return data_repaired

def imr_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    p: int = 3,
    delta: float = 1e-5,
    max_iter: int = 50
) -> np.ndarray:
    data_repaired = data_abnormal.copy().astype(np.float64)
    n_samples, n_timesteps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        y = data_repaired[sample, :, 0]
        is_abnormal = (label[sample] == 1)

        if not np.any(is_abnormal):
            continue

        y_prev = y.copy()
        for it in range(max_iter):
            # 用当前 y 构建 AR 模型（仅正常点训练）
            if n_timesteps <= p:
                break

            X = np.lib.stride_tricks.sliding_window_view(y, p)[:-1]
            y_target = y[p:]
            train_mask = ~is_abnormal[p:]

            if train_mask.sum() < p + 1:
                break

            model = LinearRegression().fit(X[train_mask], y_target[train_mask])
            y_pred = np.full(n_timesteps, np.nan)
            y_pred[p:] = model.predict(X)

            # 更新异常点
            y_new = y.copy()
            y_new[is_abnormal] = y_pred[is_abnormal]

            # 收敛检查
            diff = np.nanmax(np.abs(y_new[is_abnormal] - y_prev[is_abnormal]))
            y_prev = y_new.copy()
            y = y_new

            if diff < delta:
                break

        data_repaired[sample, :, 0] = y

    return data_repaired


# ======================================================
# ======== 3. 基于异常检测/深度模型的方法 =================
# ======================================================


# ================== imports ==================
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_argmin
import pywt
from scipy import interpolate

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用 GPU
import tensorflow as tf
from keras import layers, models, optimizers, losses


# ================== 1. DBSCAN-based repair ==================
def dbscan_repair(
    data_abnormal: np.ndarray, label: np.ndarray, eps=0.5, min_samples=5
) -> np.ndarray:
    """
    使用 DBSCAN 聚类方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        eps: DBSCAN的邻域半径
        min_samples: DBSCAN的最小样本数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask_normal = (label[sample] == 0)
        mask_abnormal = (label[sample] == 1)

        # Step 1: 检查是否有正常样本
        if not mask_normal.any():
            print(f"Warning: No normal samples found in sample {sample}. Returning original data.")
            continue

        normal_data = data_abnormal[sample, mask_normal, :]

        # Step 2: 检查缺失值
        if not np.isnan(data_abnormal[sample]).any():
            data_filled = data_abnormal[sample]
        else:
            print(f"Missing values detected in sample {sample}. Using KNNImputer trained on normal samples only.")

            # 只在正常样本上训练 KNNImputer
            n_knn = min(5, normal_data.shape[0])
            imputer = KNNImputer(n_neighbors=n_knn)
            imputer.fit(normal_data)

            # 填补整个数据集
            data_filled = imputer.transform(data_abnormal[sample])

        X = data_filled.astype(float)
        X_normal = X[mask_normal]

        # Step 3: 标准化 —— 仅使用正常样本拟合
        scaler = StandardScaler()
        X_normal_scaled = scaler.fit_transform(X_normal)
        X_scaled = scaler.transform(X)  # 整体变换

        # Step 4: 在正常样本上训练 DBSCAN
        db = DBSCAN(eps=eps, min_samples=min(min_samples, X_normal_scaled.shape[0])).fit(X_normal_scaled)
        labels_db = db.labels_  # shape: (n_normal,)

        # 提取有效簇（排除噪声 -1）
        unique_clusters = set(labels_db[labels_db != -1])

        # 计算每个簇的中心（在 scaled 空间）
        centers = {}
        for c in unique_clusters:
            centers[c] = X_normal_scaled[labels_db == c].mean(axis=0)

        global_mean_scaled = X_normal_scaled.mean(axis=0)

        # Step 5: 如果没有有效簇，使用全局正常均值修复
        if len(unique_clusters) == 0:
            fill_val = scaler.inverse_transform(global_mean_scaled.reshape(1, -1)).flatten()
            data_repaired[sample, mask_abnormal, :] = fill_val
            continue

        # 将簇中心转为数组以便计算距离
        sorted_clusters = sorted(centers.keys())
        centers_array = np.array([centers[c] for c in sorted_clusters])  # (n_clusters, n_features)

        # Step 6: 为每个异常点分配最近的簇中心（在 scaled 空间）
        anomaly_idx = np.where(mask_abnormal)[0]
        for idx in anomaly_idx:
            x_scaled = X_scaled[idx].reshape(1, -1)
            nearest_idx, _ = pairwise_distances_argmin_min(x_scaled, centers_array)
            chosen_center_scaled = centers_array[nearest_idx[0]]
            # 逆变换回原始空间
            chosen_original = scaler.inverse_transform(
                chosen_center_scaled.reshape(1, -1)
            ).flatten()
            data_repaired[sample, idx, :] = chosen_original

    return data_repaired


# ================== 2. LOF-neighbor-mean repair ==================
def lof_neighbor_mean_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_neighbors=5
) -> np.ndarray:
    """
    使用 LOF 邻近点均值方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_neighbors: 邻近点数量
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask_normal = (label[sample] == 0)
        mask_abnormal = (label[sample] == 1)

        # Step 1: 检查是否有正常样本
        if not mask_normal.any():
            print(f"Warning: No normal samples found in sample {sample}. Returning original data.")
            continue

        normal_data = data_abnormal[sample, mask_normal, :]

        # Step 2: 检查缺失值
        if not np.isnan(data_abnormal[sample]).any():
            data_filled = data_abnormal[sample]
        else:
            print(f"Missing values detected in sample {sample}. Using KNNImputer trained on normal samples only.")

            # 只在正常样本上训练 KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, normal_data.shape[0]))
            imputer.fit(normal_data)

            # 填补整个数据集
            data_filled = imputer.transform(data_abnormal[sample])

        # Step 3: 转换为 numpy 数组并确保 float 类型
        X = data_filled.astype(float)
        X_normal = X[mask_normal]

        # Step 4: 标准化 —— 仅使用正常样本拟合
        scaler = StandardScaler()
        X_normal_scaled = scaler.fit_transform(X_normal)
        X_scaled = scaler.transform(X)  # 对全部数据标准化

        # Step 5: 构建 KNN 模型（在正常样本的标准化空间中）
        n_neighbors_actual = min(n_neighbors, X_normal_scaled.shape[0])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_actual).fit(X_normal_scaled)

        # 获取正常样本的全局索引
        normal_indices_global = np.where(mask_normal)[0]

        # Step 6: 修复每个异常点
        anomaly_idx = np.where(mask_abnormal)[0]
        for idx in anomaly_idx:
            # 在标准化空间中查找最近邻（只在正常样本中）
            dist, neigh_idx_local = nbrs.kneighbors(
                X_scaled[idx].reshape(1, -1), return_distance=True
            )
            # 将局部索引映射回原始数据中的全局索引
            neigh_idx_global = normal_indices_global[neigh_idx_local[0]]
            # 在原始空间中取均值（更合理，避免逆变换误差）
            mean_vec = X[neigh_idx_global].mean(axis=0)
            data_repaired[sample, idx, :] = mean_vec

    return data_repaired


# ================== 3. Abnormal Sequence (consecutive) interpolation repair ==================
def abnormal_sequence_interpolation_repair(
    data_abnormal: np.ndarray, label: np.ndarray, method="linear"
) -> np.ndarray:
    """
    使用连续异常序列插值方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        method: 插值方法
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        n = n_timestamps
        sample_label = label[sample]
        if not (sample_label == 1).any():
            continue

        # 提取连续异常段
        segments = []
        start = None
        for i in range(n):
            if sample_label[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None
        if start is not None:
            segments.append((start, n - 1))

        x_full = np.arange(n)
        for col in range(n_features):
            y = data_abnormal[sample, :, col].astype(float)
            # 将异常点设为 NaN，确保插值时不被误用
            y_clean = y.copy()
            y_clean[sample_label == 1] = np.nan

            for s, e in segments:
                # 查找左右最近的正常点
                left = s - 1
                while left >= 0 and sample_label[left] == 1:
                    left -= 1
                right = e + 1
                while right < n and sample_label[right] == 1:
                    right += 1

                if left >= 0 and right < n:
                    # 两端都有正常点 → 插值
                    xp = [left, right]
                    fp = [y_clean[left], y_clean[right]]
                    f = interpolate.interp1d(xp, fp, kind=method, fill_value="extrapolate")
                    y[s : e + 1] = f(np.arange(s, e + 1))
                elif left >= 0:
                    y[s : e + 1] = y_clean[left]
                elif right < n:
                    y[s : e + 1] = y_clean[right]
                # else: no normal points at all — leave as is

            data_repaired[sample, :, col] = y
            
    return data_repaired


# ================== 4. Window-based regression repair ==================
def window_regression_repair(
    data_abnormal: np.ndarray, label: np.ndarray, window=5
) -> np.ndarray:
    """
    使用基于窗口的回归方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: 窗口大小
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        X_full = data_abnormal[sample].astype(float)
        anomaly_idx = np.where(label[sample] == 1)[0]

        for t in anomaly_idx:
            start = max(0, t - window)
            end = min(n_timestamps, t + window + 1)
            local_idx = np.arange(start, end)
            train_idx = local_idx[
                (local_idx != t) & (label[sample, local_idx] == 0)
            ]  # 排除 t 自身，且只取正常点

            if len(train_idx) < max(5, n_features):
                # 回退到列插值
                for col in range(n_features):
                    col_data = data_repaired[sample, :, col].astype(float)
                    nan_mask = np.zeros(n_timestamps, dtype=bool)
                    nan_mask[anomaly_idx] = True
                    col_data[nan_mask] = np.nan
                    col_data = pd.Series(col_data).interpolate(
                        method="linear", limit_direction="both"
                    )
                    data_repaired[sample, :, col] = col_data
                continue

            # 构造训练特征：包括相对时间 + 所有变量
            X_train, Y_train = [], []
            for i in train_idx:
                rel_time = i - t  # 相对目标时刻的时间偏移
                feat = np.concatenate([[rel_time], X_full[i]])
                X_train.append(feat)
                Y_train.append(X_full[i])  # 预测完整向量

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            # 多输出回归
            model = LinearRegression().fit(X_train, Y_train)

            # 构造测试特征：相对时间为 0，其他变量用邻域正常样本均值
            neighbor_mean = X_full[train_idx].mean(axis=0)
            test_feat = np.concatenate([[0], neighbor_mean]).reshape(1, -1)
            pred = model.predict(test_feat)[0]
            data_repaired[sample, t, :] = pred

    return data_repaired


# ================== 5. Clustering (KMeans) repair ==================
def clustering_kmeans_repair(
    data_abnormal: np.ndarray, label: np.ndarray, n_clusters=5
) -> np.ndarray:
    """
    使用 KMeans 聚类方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        n_clusters: 聚类数量
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        mask_normal = (label[sample] == 0)
        mask_abnormal = (label[sample] == 1)

        # Step 1: 检查是否有正常样本
        if not mask_normal.any():
            print(f"Warning: No normal samples found in sample {sample}. Returning original data.")
            continue

        normal_data = data_abnormal[sample, mask_normal, :]

        # Step 2: 检查是否存在缺失值
        if not np.isnan(data_abnormal[sample]).any():
            # 无缺失值，直接使用原始数据
            data_filled = data_abnormal[sample]
        else:
            print(f"Missing values detected in sample {sample}. Using KNNImputer trained on normal samples only.")

            # ✅ 关键：只用正常样本拟合 KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, normal_data.shape[0]))
            imputer.fit(normal_data)  # 只在正常样本上 fit

            # 对整个数据集进行填补
            data_filled = imputer.transform(data_abnormal[sample])

        # 确保为 float 类型
        X = data_filled.astype(float)
        normal_X = X[mask_normal]

        # Step 3: 如果正常样本太少，无法聚类
        if normal_X.shape[0] < n_clusters:
            center = normal_X.mean(axis=0)
            data_repaired[sample, mask_abnormal, :] = center
            continue

        # Step 4: 在正常样本上训练 KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(normal_X)
        centers = kmeans.cluster_centers_

        # Step 5: 标准化（同样只用正常样本拟合 scaler）
        scaler = StandardScaler().fit(normal_X)
        X_scaled = scaler.transform(X)
        centers_scaled = scaler.transform(centers)

        # Step 6: 为异常点分配最近的簇中心
        anomaly_idx = np.where(mask_abnormal)[0]
        if len(anomaly_idx) > 0:
            closest_center_ids = pairwise_distances_argmin(
                X_scaled[anomaly_idx], centers_scaled
            )
            repaired_values = centers[closest_center_ids]  # 使用原始空间的中心值
            data_repaired[sample, anomaly_idx, :] = repaired_values

    return data_repaired


# ================== 6. Wavelet-based repair ==================
def wavelet_denoise_repair(
    data_abnormal: np.ndarray, label: np.ndarray, wavelet="db4", level=None
) -> np.ndarray:
    """
    使用小波去噪方法修复异常值
    
    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        wavelet: 小波基函数
        level: 小波分解层数
        
    返回:
        修复后的三维numpy数组
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        n = n_timestamps
        for col in range(n_features):
            series = data_abnormal[sample, :, col].astype(float)
            # fill NaNs for transform
            series_filled = (
                pd.Series(series)
                .interpolate()
                .fillna(method="ffill")
                .fillna(method="bfill")
                .values
            )
            # compute wavelet coeffs
            coeffs = pywt.wavedec(series_filled, wavelet=wavelet, level=level)
            # thresholding (VisuShrink-like): universal threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0
            uthresh = sigma * np.sqrt(2 * np.log(len(series_filled))) if sigma > 0 else 0
            denoised_coeffs = [
                pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs
            ]
            reconstructed = pywt.waverec(denoised_coeffs, wavelet)
            # ensure length match
            if len(reconstructed) > n:
                reconstructed = reconstructed[:n]
            elif len(reconstructed) < n:
                reconstructed = np.pad(
                    reconstructed, (0, n - len(reconstructed)), mode="edge"
                )
            series_out = series.copy()
            series_out[label[sample] == 1] = reconstructed[label[sample] == 1]
            data_repaired[sample, :, col] = series_out
            
    return data_repaired


# ================== 7. LSTM-based sequence prediction repair ==================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

def lstm_sequence_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window=5,
    epochs=20,
    batch_size=8,
):
    """
    使用 LSTM 修复异常值（GPU 加速 PyTorch 版本）

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: int, 滑动窗口大小
        epochs: int, 训练轮数
        batch_size: int, 批次大小

    返回:
        修复后的三维numpy数组
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    import warnings

    # ================================
    # 设备设置：优先使用 GPU
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm_sequence_repair] Using device: {device}")

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = (label[sample] == 0)
        mask_anomaly = (label[sample] == 1)

        # -----------------------------
        # 1. 提取正常样本
        # -----------------------------
        X_normal = data_abnormal[sample, mask_normal, :].astype("float32")
        n_normal_samples, _ = X_normal.shape

        if n_normal_samples <= window:
            fill_value = X_normal.mean(axis=0) if n_normal_samples > 0 else data_abnormal[sample].mean(axis=0)
            data_repaired[sample, mask_anomaly, :] = fill_value
            continue

        # -----------------------------
        # 2. 构建滑动窗口训练集
        # -----------------------------
        X_train = []
        y_train = []
        for i in range(n_normal_samples - window):
            X_train.append(X_normal[i:i + window])
            y_train.append(X_normal[i + window])
        X_train = np.array(X_train)  # (N, window, n_features)
        y_train = np.array(y_train)  # (N, n_features)

        X_train = torch.tensor(X_train).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)

        # -----------------------------
        # 3. 定义 LSTM 模型
        # -----------------------------
        class SequenceLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=32):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                _, (h, _) = self.lstm(x)  # 取最后一个隐状态
                out = self.fc(h[-1])      # (batch, hidden) -> (batch, input_dim)
                return out

        model = SequenceLSTM(n_features, hidden_dim=32).to(device)

        try:
            model = torch.compile(model)
            print(f"[lstm_sequence_repair] Model compiled for faster training (sample {sample}).")
        except Exception as e:
            print(f"[lstm_sequence_repair] Compile failed: {e}")

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 混合精度训练（仅 CUDA）
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # ==============================
        # 4. 训练 LSTM
        # ==============================
        model.train()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()

                if device.type == 'cuda' and scaler is not None:
                    with torch.autocast(device_type='cuda'):
                        pred = model(x_batch)
                        loss = criterion(pred, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()

        # ==============================
        # 5. 修复异常点
        # ==============================
        model.eval()
        X_raw = data_abnormal[sample].astype("float32")  # 原始数据用于上下文检查

        with torch.no_grad():
            for idx in range(n_timestamps):
                if not mask_anomaly[idx]:
                    continue  # 只处理异常点

                start_idx = idx - window
                if start_idx < 0:
                    # 不够 window 长度，用正常样本均值填充
                    fill_value = X_normal.mean(axis=0)
                    data_repaired[sample, idx, :] = fill_value
                    continue

                # 检查上下文 [start_idx, idx) 是否全为正常点
                context_labels = label[sample, start_idx:idx]
                if (context_labels == 1).any():
                    # 上下文有异常：使用最后 window 个正常点
                    normal_indices = np.where(mask_normal)[0]
                    if len(normal_indices) >= window:
                        last_normal_indices = normal_indices[-window:]
                        input_seq = X_raw[last_normal_indices, :]
                    else:
                        mean_val = X_normal.mean(axis=0)
                        input_seq = np.tile(mean_val, (window, 1)).astype("float32")
                else:
                    # 上下文干净，直接使用原始数据
                    input_seq = X_raw[start_idx:idx]

                # 转为 tensor 并预测
                input_tensor = torch.tensor(input_seq).float().unsqueeze(0).to(device)  # (1, window, n_features)
                pred = model(input_tensor).cpu().numpy()[0]  # (n_features,)
                data_repaired[sample, idx, :] = pred

    return data_repaired


# ================== 8. GAN-based repair (generator trained on normal samples) ==================
def gan_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    noise_dim=10,
    epochs=50,
    batch_size=8,
):
    """
    使用 GAN 修复异常值（PyTorch + GPU 安全版本）
    使用 BCEWithLogitsLoss 避免 autocast 下的数值不稳定

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        noise_dim: 噪声维度
        epochs: 训练轮数
        batch_size: 批次大小

    返回:
        修复后的三维numpy数组
    """

    # ================================
    # 设备设置：优先使用 GPU
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gan_repair] Using device: {device}")

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = label[sample] == 0
        mask_anomaly = label[sample] == 1

        # -----------------------------
        # 1. 提取正常样本并处理缺失值
        # -----------------------------
        X_normal_raw = data_abnormal[sample, mask_normal, :]
        if X_normal_raw.shape[0] == 0:
            warnings.warn(f"[GAN-Repair] No normal samples found in sample {sample}.")
            continue

        if np.isnan(X_normal_raw).any():
            n_neighbors = min(5, X_normal_raw.shape[0])
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X_normal = imputer.fit_transform(X_normal_raw).astype("float32")
        else:
            X_normal = X_normal_raw.astype("float32")

        X_normal = torch.tensor(X_normal).float().to(device)

        if X_normal.shape[0] < 5:
            fill_value = X_normal.mean(dim=0).cpu().numpy()
            data_repaired[sample, mask_anomaly, :] = fill_value
            continue

        # -----------------------------
        # 2. 定义生成器与判别器
        # -----------------------------
        class Generator(nn.Module):
            def __init__(self, noise_dim, feat_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(noise_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, feat_dim)
                )

            def forward(self, z):
                return self.net(z)

        class Discriminator(nn.Module):
            def __init__(self, feat_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(feat_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1)
                    # ✅ 移除 Sigmoid！由 BCEWithLogitsLoss 自动处理
                )

            def forward(self, x):
                return self.net(x)

        generator = Generator(noise_dim, n_features).to(device)
        discriminator = Discriminator(n_features).to(device)

        try:
            generator = torch.compile(generator)
            discriminator = torch.compile(discriminator)
            print(f"[gan_repair] Models compiled for faster training (sample {sample}).")
        except Exception as e:
            print(f"[gan_repair] Compile failed: {e}")

        # 优化器
        opt_g = optim.Adam(generator.parameters(), lr=0.001)
        opt_d = optim.Adam(discriminator.parameters(), lr=0.001)

        # ✅ 使用 BCEWithLogitsLoss（安全支持 autocast）
        criterion = nn.BCEWithLogitsLoss()

        # 混合精度训练（仅 CUDA）
        scaler_d = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        scaler_g = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # ================================
        # 3. 训练 GAN
        # ================================
        generator.train()
        discriminator.train()

        # 固定标签（不需要梯度）
        real_labels = torch.ones(batch_size, 1).to(device)      # 真实样本标签
        fake_labels = torch.zeros(batch_size, 1).to(device)      # 伪造样本标签

        for epoch in range(epochs):
            for _ in range(X_normal.shape[0] // batch_size):
                # -----------------------------
                # 训练判别器
                # -----------------------------
                opt_d.zero_grad()

                # 真实样本
                idx = torch.randint(0, X_normal.shape[0], (batch_size,))
                real_samples = X_normal[idx]

                # 生成假样本
                noise = torch.randn(batch_size, noise_dim).to(device)
                with torch.no_grad():
                    fake_samples = generator(noise)

                # 拼接输入
                combined = torch.cat([real_samples, fake_samples], dim=0)
                labels = torch.cat([real_labels, fake_labels], dim=0)

                if device.type == 'cuda' and scaler_d is not None:
                    with torch.autocast(device_type='cuda'):
                        logits = discriminator(combined)
                        d_loss = criterion(logits, labels)
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
                else:
                    logits = discriminator(combined)
                    d_loss = criterion(logits, labels)
                    d_loss.backward()
                    opt_d.step()

                # -----------------------------
                # 训练生成器
                # -----------------------------
                opt_g.zero_grad()
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_samples = generator(noise)
                logits = discriminator(fake_samples)

                if device.type == 'cuda' and scaler_g is not None:
                    with torch.autocast(device_type='cuda'):
                        g_loss = criterion(logits, real_labels)  # 希望被判别器认为是真实的
                    scaler_g.scale(g_loss).backward()
                    scaler_g.step(opt_g)
                    scaler_g.update()
                else:
                    g_loss = criterion(logits, real_labels)
                    g_loss.backward()
                    opt_g.step()

        # ================================
        # 4. 修复异常点
        # ================================
        generator.eval()
        n_anomaly = mask_anomaly.sum()
        if n_anomaly > 0:
            with torch.no_grad():
                noise = torch.randn(n_anomaly, noise_dim).to(device)
                if device.type == 'cuda' and scaler_g is not None:
                    with torch.autocast(device_type='cuda'):
                        generated = generator(noise).cpu().numpy()
                else:
                    generated = generator(noise).cpu().numpy()
            anomaly_indices = np.where(mask_anomaly)[0]
            data_repaired[sample, anomaly_indices, :] = generated

    return data_repaired


# ================== 9. TranAD-based repair  ==================
def tranad_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window=5,
    epochs=10,
    batch_size=32,
    lr=0.001,
):
    """
    使用 MLP 预测器修复异常值（支持 GPU/CUDA）

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: int, 滑动窗口长度（上下文长度）
        epochs: int, 训练轮数
        batch_size: int, 批次大小
        lr: float, 学习率

    返回:
        修复后的三维numpy数组
    """


    # ================================
    # 设备自动选择 (GPU > CPU)
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tranad_repair] Using device: {device}")

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = label[sample] == 0
        mask_anomaly = label[sample] == 1

        # -----------------------------
        # 1. 提取正常样本并处理缺失值
        # -----------------------------
        X_normal_raw = data_abnormal[sample, mask_normal, :]
        if X_normal_raw.shape[0] == 0:
            warnings.warn("[TranAD] No normal samples found.")
            continue

        if np.isnan(X_normal_raw).any():
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, X_normal_raw.shape[0]))
            X_normal = imputer.fit_transform(X_normal_raw).astype("float32")
        else:
            X_normal = X_normal_raw.astype("float32")

        if X_normal.shape[1] != n_features:
            n_features = X_normal.shape[1]

        if X_normal.shape[0] <= window:
            mean_val = torch.tensor(X_normal.mean(axis=0)).float().to(device)
            data_repaired[sample, mask_anomaly, :] = mean_val.cpu().numpy()
            continue

        # -----------------------------
        # 2. 构建滑动窗口数据
        # -----------------------------
        X_train = np.array([
            X_normal[i : i + window] for i in range(X_normal.shape[0] - window)
        ])
        X_train_tensor = torch.tensor(X_train).float().to(device)  # 移到 GPU
        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # -----------------------------
        # 3. 定义模型：MLP Reconstructor
        # -----------------------------
        class MLPReconstructor(nn.Module):
            def __init__(self, L, F):
                super().__init__()
                hidden_dim = 64
                self.net = nn.Sequential(
                    nn.Linear(L * F, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, F)
                )

            def forward(self, x):
                B, L, F = x.shape
                x_flat = x.view(B, -1)  # (B, L*F)
                return self.net(x_flat)  # (B, F)

        model = MLPReconstructor(window, n_features).to(device)

        try:
            model = torch.compile(model)
            print(f"[tranad_repair] Model compiled for faster training (sample {sample}).")
        except Exception as e:
            print(f"[tranad_repair] Compile failed: {e}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 混合精度训练（仅 CUDA）
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # -----------------------------
        # 4. 训练模型
        # -----------------------------
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                # batch 已在 __init__ 中移到 GPU
                target = batch[:, -1, :]  # 预测最后一个时间步

                optimizer.zero_grad()

                if device.type == 'cuda' and scaler is not None:
                    with torch.autocast(device_type='cuda'):
                        pred = model(batch)
                        loss = criterion(pred, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(batch)
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        # -----------------------------
        # 5. 修复异常点（批量推理）
        # -----------------------------
        model.eval()
        anomaly_indices = np.where(mask_anomaly)[0]
        contexts = []
        positions = []

        for idx in anomaly_indices:
            if idx >= window:
                ctx = data_repaired[sample, idx - window : idx, :].astype("float32")
                contexts.append(ctx)
                positions.append(idx)

        if contexts:
            # 转为 tensor 并移到设备
            contexts = torch.tensor(np.array(contexts)).float().to(device)
            with torch.no_grad():
                if device.type == 'cuda' and scaler is not None:
                    with torch.autocast(device_type='cuda'):
                        repairs = model(contexts)
                else:
                    repairs = model(contexts)
                repairs = repairs.cpu().numpy()  # 移回 CPU

            # 写入修复值
            for pos, val in zip(positions, repairs):
                data_repaired[sample, pos, :] = val

    return data_repaired


# ================== 10. IMDiffusion-based repair  ==================
import torch.optim as optim

def imdiffusion_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    timesteps=50,
    epochs=20,
    batch_size=8,
    lr=0.001,
    mask_ratio=0.15,
):
    """
    使用扩散模型修复异常值

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        timesteps: 扩散时间步数
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        mask_ratio: 模拟异常时的掩码比例

    返回:
        修复后的三维numpy数组
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[imdiffusion_repair] Using device: {device}")

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = label[sample] == 0
        mask_anomaly = label[sample] == 1

        X_normal_raw = data_abnormal[sample, mask_normal, :]
        if X_normal_raw.shape[0] == 0:
            warnings.warn("[IMDiffusion] No normal samples found.")
            continue

        # KNN 填补原始缺失
        if np.isnan(X_normal_raw).any():
            imputer = KNNImputer(n_neighbors=min(5, X_normal_raw.shape[0]))
            X_normal_imputed = imputer.fit_transform(X_normal_raw)
        else:
            X_normal_imputed = X_normal_raw

        # 标准化
        scaler = StandardScaler()
        X_normal_scaled = scaler.fit_transform(X_normal_imputed)
        X_normal = torch.tensor(X_normal_scaled).float().to(device)

        if X_normal.shape[0] < 5:
            data_repaired[sample, mask_anomaly, :] = scaler.inverse_transform(
                X_normal.mean(dim=0).cpu().numpy().reshape(1, -1)
            ).squeeze()
            continue

        # 扩散参数
        beta = torch.linspace(0.0001, 0.02, timesteps).to(device)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        def q_sample(x_0, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_0)
            sqrt_alpha_bar_t = torch.gather(alpha_bar, 0, t).sqrt().unsqueeze(1)
            sqrt_one_minus_alpha_bar_t = (1 - torch.gather(alpha_bar, 0, t)).sqrt().unsqueeze(1)
            return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        # 时间嵌入
        class SinusoidalPositionEmbeddings(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            def forward(self, time):
                device = time.device
                half_dim = self.dim // 2
                embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
                embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
                embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
                embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
                return embeddings

        class DiffusionNet(nn.Module):
            def __init__(self, feat_dim, t_emb_dim=16):
                super().__init__()
                self.time_mlp = nn.Sequential(
                    SinusoidalPositionEmbeddings(t_emb_dim),
                    nn.Linear(t_emb_dim, 32),
                    nn.ReLU(),
                )
                self.net = nn.Sequential(
                    nn.Linear(feat_dim + 32, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, feat_dim),
                )
            def forward(self, x, t):
                t_emb = self.time_mlp(t)
                x_in = torch.cat([x, t_emb], dim=1)
                return self.net(x_in)

        model = DiffusionNet(n_features).to(device)
        try:
            model = torch.compile(model)
            print(f"[imdiffusion_repair] Model compiled (sample {sample}).")
        except:
            pass

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scaler_amp = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # 训练：预测噪声
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            indices = torch.randperm(X_normal.shape[0], device=device)
            for i in range(0, indices.shape[0], batch_size):
                batch_idx = indices[i : i + batch_size]
                x_0 = X_normal[batch_idx]

                # 模拟异常
                mask = torch.rand_like(x_0) < mask_ratio
                x_corrupted = x_0.clone()
                x_corrupted[mask] = 0.0

                t = torch.randint(0, timesteps, (x_0.shape[0],), device=device)
                noise = torch.randn_like(x_0)
                x_t = q_sample(x_corrupted, t, noise)

                optimizer.zero_grad()
                if device.type == 'cuda' and scaler_amp is not None:
                    with torch.autocast(device_type='cuda'):
                        pred_noise = model(x_t, t)
                        loss = criterion(pred_noise, noise)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    pred_noise = model(x_t, t)
                    loss = criterion(pred_noise, noise)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()

        # 修复：使用原始异常数据加噪后去噪
        model.eval()
        anomaly_indices = np.where(mask_anomaly)[0]
        if len(anomaly_indices) == 0:
            continue

        with torch.no_grad():
            x_abnormal = torch.tensor(data_abnormal[sample, mask_anomaly, :]).float().to(device)
            # 标准化
            x_abnormal = torch.tensor(scaler.transform(x_abnormal.cpu().numpy())).float().to(device)
            # 加噪到 T
            t_full = torch.full((x_abnormal.shape[0],), timesteps-1, device=device)
            x = q_sample(x_abnormal, t_full)

            # 反向去噪（预测噪声）
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((x.shape[0],), t, device=device)
                pred_noise = model(x, t_tensor)
                z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

                x = (1 / alpha[t].sqrt()) * (x - (1 - alpha[t]) / (1 - alpha_bar[t]).sqrt() * pred_noise)
                x = x + beta[t].sqrt() * z

            # 逆标准化
            repaired_values = scaler.inverse_transform(x.cpu().numpy())
            data_repaired[sample, anomaly_indices, :] = repaired_values

    return data_repaired

# ================== 11. CAE-M repair  ==================
def cae_m_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window: int = 10,  # 现在支持任意长度！
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    latent_dim: int = 8,
) -> np.ndarray:
    """
    基于 MLP-Autoencoder 的异常修复方法（支持任意 window + CPU/GPU）
    完全避免卷积池化导致的长度不匹配问题

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window: 窗口大小
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        latent_dim: 潜在空间维度

    返回:
        修复后的三维numpy数组
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import pandas as pd
    import warnings

    # ================================
    # 设备设置
    # ================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cae_m_repair] Using device: {device}")

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        mask_normal = label[sample] == 0
        mask_anomaly = label[sample] == 1

        X_normal_raw = data_abnormal[sample, mask_normal, :]
        if X_normal_raw.shape[0] == 0:
            warnings.warn("[CAE-M] No normal samples found.")
            continue

        # 缺失值处理
        if np.isnan(X_normal_raw).any():
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=min(5, X_normal_raw.shape[0]))
            X_normal = imputer.fit_transform(X_normal_raw).astype("float32")
        else:
            X_normal = X_normal_raw.astype("float32")

        if X_normal.shape[1] != n_features:
            n_features = X_normal.shape[1]

        if X_normal.shape[0] <= window:
            mean_val = X_normal.mean(axis=0)
            data_repaired[sample, mask_anomaly, :] = mean_val
            continue

        # 构建窗口
        X_train = np.array([
            X_normal[i : i + window] for i in range(X_normal.shape[0] - window)
        ])
        X_train_tensor = torch.tensor(X_train).float()
        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ================================
        # MLP-Autoencoder：支持任意 window
        # ================================
        class MLPAE(nn.Module):
            def __init__(self, seq_len, n_features, latent_dim):
                super(MLPAE, self).__init__()
                self.seq_len = seq_len
                self.n_features = n_features
                self.latent_dim = latent_dim

                # 将整个窗口展平
                self.input_dim = seq_len * n_features

                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim),
                    nn.ReLU()
                )

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.input_dim),
                    nn.Sigmoid()  # 可改为 Identity + 标准化
                )

            def forward(self, x):
                # x: (B, L, F)
                B, L, F = x.shape
                x_flat = x.view(B, -1)  # (B, L*F)
                encoded = self.encoder(x_flat)
                decoded_flat = self.decoder(encoded)
                decoded = decoded_flat.view(B, L, F)
                return decoded

        model = MLPAE(seq_len=window, n_features=n_features, latent_dim=latent_dim).to(device)

        try:
            model = torch.compile(model)
            print(f"[cae_m_repair] Model compiled (sample {sample}).")
        except Exception as e:
            print(f"[cae_m_repair] Compile failed: {e}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 混合精度（仅 CUDA）
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        # 训练
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                if device.type == 'cuda' and scaler is not None:
                    with torch.autocast(device_type='cuda'):
                        output = model(batch)
                        loss = criterion(output, batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch)
                    loss = criterion(output, batch)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        # ================================
        # 修复异常点
        # ================================
        model.eval()
        anomaly_indices = np.where(mask_anomaly)[0]
        contexts = []
        positions = []

        for idx in anomaly_indices:
            if idx >= window:
                ctx = data_repaired[sample, idx - window : idx, :].astype("float32")
                contexts.append(ctx)
                positions.append(idx)

        if contexts:
            contexts = torch.tensor(np.array(contexts)).float().to(device)
            with torch.no_grad():
                if device.type == 'cuda' and scaler is not None:
                    with torch.autocast(device_type='cuda'):
                        repairs = model(contexts)
                else:
                    repairs = model(contexts)
                # 取最后一个时间步
                repairs = repairs[:, -1, :].cpu().numpy()

            for pos, val in zip(positions, repairs):
                data_repaired[sample, pos, :] = val

    return data_repaired


# ================== 12. Akane repair  ==================
from Error_Cleaner.tools.outlier_modifiers.akane0 import MultivariateCleaner, MVPatternMiner, MVRepairer, PerplexityModel
def akane_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window_len: int = 40, 
    k_range: tuple = (5, 15), 
    markov_order: int = 2,
    repair_method: str = 'cubic_spline'
) -> np.ndarray:
    """
    基于 Akane 困惑度指导的多变量时序数据清洗方法。
    强制对所有 label=1 的数据点进行值替换。

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        window_len: 窗口长度
        k_range: k范围
        markov_order: 马尔可夫阶数
        repair_method: 修复方法

    返回:
        修复后的三维numpy数组
    """
    
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    for sample in range(n_samples):
        # --- 1. 数据准备 ---
        X = data_abnormal[sample].astype(float)
        
        # 确定需要清洗的候选点索引 (label=1)
        mask_anomaly = label[sample] == 1
        candidate_indices = np.where(mask_anomaly)[0]
        
        num_labeled_points = len(candidate_indices)
        
        if num_labeled_points == 0:
            print(f"没有找到需要清洗的候选点 (label=1) in sample {sample}。数据未修改。")
            continue
        
        print(f"检测到 {num_labeled_points} 个必须修复的异常点 (label=1) in sample {sample}。")

        # --- 2. 预插值 NaN ---
        X_filled = pd.DataFrame(X).interpolate().bfill().ffill().values 

        # --- 3. 模式发现 (MVPatternMiner) ---
        miner = MVPatternMiner(window_len=window_len, stride=5)
        miner.auto_fit(X_filled, k_range=k_range) 
        
        N_PATTERNS_AUTO = miner.n_patterns
        
        # --- 4. 修复方法选择 ---
        if repair_method == 'cubic_spline':
            mv_repairer = lambda X, i: MVRepairer.cubic_spline(X, i, window=10)
        else:
            mv_repairer = lambda X, i: MVRepairer.local_linear(X, i, window=10)

        # --- 5. 清洗器初始化与执行 ---
        # 预算参数不再用于约束，但 Cleaner 内部逻辑已修改为修复所有 label=1 的点
        cleaner = MultivariateCleaner(
            backend='markov', 
            n_components=N_PATTERNS_AUTO, 
            markov_order=markov_order,
            repairer=mv_repairer,
            miner=miner,
            candidate_indices=candidate_indices
        )
        
        # 传入的 budget 只是一个形式参数，不参与约束逻辑
        X_cleaned, selected_repairs, final_ppl = cleaner.greedy_clean(X_filled, budget=num_labeled_points)
        
        # --- 6. 结果封装 ---
        data_repaired[sample, :, :] = X_cleaned
        
        print(f"\n--- 多变量 Akane 清洗完成 (sample {sample}) ---")
        print(f"原始困惑度 (PPL): {cleaner.initial_ppl:.4f}")
        print(f"最终困惑度 (PPL): {final_ppl:.4f}")
        print(f"总修复点数: {len(selected_repairs)}")
    
    return data_repaired


def ols_residual_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    p: int = 3,
    max_iter: int = 100
) -> np.ndarray:
    """
    使用OLS残差方法修复异常值

    参数:
        data_abnormal: 三维numpy数组，形状为 (n_samples, n_timestamps, n_features)
        label: 二维numpy数组，形状为 (n_samples, n_timestamps)，标记异常点（1表示异常，0表示正常）
        p: 阶数
        max_iter: 最大迭代次数

    返回:
        修复后的三维numpy数组
    """
    import numpy as np
    from scipy.linalg import lstsq

    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    for sample in range(n_samples):
        values = data_abnormal[sample].astype(float)
        anomaly_mask = label[sample].astype(bool)

        repaired = values.copy()
        if not np.any(anomaly_mask):
            continue

        for iter_idx in range(max_iter):
            changes_made = False
            for col in range(n_features):
                series = repaired[:, col]
                for t in range(1, n_timestamps):  # 从 t=1 开始，t=0 无法预测
                    if not anomaly_mask[t]:
                        continue

                    # 构建训练集：使用 t>=1 且非当前点
                    X_hist, y_hist = [], []
                    for i in range(1, n_timestamps):
                        if i == t:
                            continue
                        start = max(0, i - p)
                        x_row = series[start:i][::-1]
                        y_row = series[i]
                        X_hist.append(np.pad(x_row, (p - len(x_row), 0), mode='edge'))
                        y_hist.append(y_row)

                    if len(X_hist) < p:
                        continue

                    X_hist = np.array(X_hist)
                    y_hist = np.array(y_hist)

                    try:
                        phi, *rest = lstsq(X_hist, y_hist)  # (p,)
                    except:
                        continue

                    # 预测 t
                    start_t = max(0, t - p)
                    x_window = series[start_t:t][::-1]
                    x_pred = np.pad(x_window, (p - len(x_window), 0), mode='edge')
                    pred = x_pred @ phi
                    current = repaired[t, col]
                    repaired[t, col] = pred
                    if abs(pred - current) > 1e-8:
                        changes_made = True

            if not changes_made:
                break
                
        data_repaired[sample] = repaired

    return data_repaired


    return pd.DataFrame(repaired, columns=data_abnormal.columns, index=data_abnormal.index)


if __name__ == "__main__":

    from Error_Injection.injector import DataManager
    from Datasets.load_dataset import load_single_dataset

    type = 'forecast'
    dataset_name = 'ETTh1'
    data, label =load_single_dataset(type, dataset_name)

    if type == 'forecast':
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        print(data.shape)
        dm = DataManager(data, abnormal_rate=0.1, task_type=type)
    else:
        print(data.shape, label.shape)
        dm = DataManager(data, label, abnormal_rate=0.1, task_type=type)
    dm.inject_errors(
        0.3,
        ["single", "drift", "gaussian", "volatility", "gradual", "sudden"],
        covered_attrs=dm.clean_data.columns,
    )

    dirty_data, error_mask = dm.get_dirty_data_restored()
    data_abnormal = dirty_data
    label = dm.error_mask.any(axis=1).astype(int)
    print(label.unique())

    # methods = [
    # moving_average_repair,
    # autoregressive_repair,
    # arma_repair,
    # kalman_filter_repair,
    # interpolation_repair,
    # state_space_repair,
    # trajectory_simplification_repair,
    # exponential_smoothing_repair,
    # svr_repair,

    # maximum_likelihood_repair,
    # bayesian_model_repair,
    # markov_model_repair,
    # hmm_repair,
    # smurf_repair,
    # spatio_temporal_prob_model_repair,
    # em_repair,
    # relationship_network_repair,
    # gaussian_mixture_repair,
    # imr_repair,

    # dbscan_repair,
    # lof_neighbor_mean_repair,
    # abnormal_sequence_interpolation_repair,
    # window_regression_repair,
    # clustering_kmeans_repair,
    # wavelet_denoise_repair,
    # lstm_sequence_repair,
    # gan_repair,
    # tranad_repair,
    # imdiffusion_repair,
    # cae_m_repair,
    # akane_repair,

    # ]

    methods = [ols_residual_repair,]


    results = {}
    for func in methods:
        print(f"\n 运行 {func.__name__} ...")
        try:
            data_repaired = func(data_abnormal, label)
            if not isinstance(data_repaired, pd.DataFrame):
                raise ValueError("返回类型错误，必须是 DataFrame")
            # 简单的效果验证：修复点的数值波动应减小
            diff = np.mean(np.abs(data_repaired - data_abnormal))
            results[func.__name__] = diff
            print(f"{func.__name__} 完成，平均改变量: {diff:.4f}")
        except Exception as e:
            print(f"{func.__name__} 失败: {e}")
            results[func.__name__] = None

    print("\n=== 测试结果汇总 ===")
    for name, score in results.items():
        print(f"{name:35s} => {'Fail' if score is None else f'{score:.4f}'}")
