import pandas as pd
import numpy as np
import time
import pickle
import os

from Datasets.load_dataset import load_single_dataset, sample_data_by_rate, convert_to_unix_timestamp
from Error_Injection.injector import DataManager

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from aeon.transformations.collection.convolution_based import MiniRocket
from DLiner import DLinear
from LSTMForecast import LSTMForecast

from sklearn.metrics import (
    roc_auc_score, f1_score, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score,
    mean_squared_error
)


# 添加常量定义
window_size_f = 50
horizon_f = 30
window_step = 10
K_MIN, K_MAX = 2, 30

def load_original_data(dataset_name, dataset_type, rate=1):
    original_dataset_dir = "/home/yyy/TSC/TSClean/AutoClean/Datasets/original"
    if dataset_type == 'clean':
        csv_file = f"{original_dataset_dir}/{dataset_name}/{dataset_name}_Clean.csv"
    elif dataset_type == 'dirty':
        csv_file = f"{original_dataset_dir}/{dataset_name}/{dataset_name}_Dirty.csv"

    try:
        df = pd.read_csv(csv_file)
        
        # 根据rate获取前rate%的数据
        if rate < 1:
            n_rows = len(df)
            n_selected = max(1, int(n_rows * rate))
            # 对于时间序列数据，我们使用前n_selected行以保持时间顺序
            df = df.iloc[:n_selected]
        
        # 检查并转换时间列
        for col in df.columns:
            # 检查是否是时间相关的列（第一列通常为时间戳）
            if col.lower() in ['timestamp', 'time', 'date', 'datetime']:
                df[col] = convert_to_unix_timestamp(df[col])
            # 即使列名不明显是时间列，也尝试检测是否包含时间字符串
            else:
                # 取样前几个非空值进行检测
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0 and all(isinstance(val, str) for val in sample_values):
                    # 检查是否可能是日期时间格式
                    try:
                        pd.to_datetime(sample_values, errors='raise')
                        df[col] = convert_to_unix_timestamp(df[col])
                    except (ValueError, TypeError, OverflowError):
                        # 不是时间格式，跳过
                        pass
        
        data = df.values
        
        # 转换为(1, n_timestamps, n_features)格式
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        return data
    
    except Exception as e:
        raise RuntimeError(f"加载 {csv_file} 时出错: {e}")

def to_supervised(data_3d, window_size, horizon):
    """
    将单条多变量时间序列 (1, T, F) 转换为监督学习格式 (N, window_size, F)
    
    Parameters:
    ----------
    data_3d : np.ndarray
        形状为 (1, n_timesteps, n_features) 的输入数据
    window_size : int
        历史窗口长度（用于预测）
    horizon : int
        预测步长

    Returns:
    -------
    X : np.ndarray of shape (N, window_size, n_features)
        输入特征
    y : np.ndarray of shape (N, horizon, n_features)
        目标值（下一个时间步的真实值）
    """
    if data_3d.shape[0] != 1:
        raise ValueError("Forecast task expects sample size = 1")
    
    series = data_3d.squeeze(0)  # (T, F)
    T, F = series.shape
    
    if T <= window_size:
        # 数据太少，无法构造窗口
        return np.empty((0, window_size, F)), np.empty((0, horizon, F))
    
    X, y = [], []

    for t in range(0, T - window_size - horizon, window_step):
        X_window = series[t : t + window_size]                   # (window_size, F)
        Y_future = series[t + window_size : t + window_size + horizon]  # (horizon, F)

        X.append(X_window)
        y.append(Y_future)
    
    return np.array(X), np.array(y)  # (N, W, F), (N, H, F)


def supervised_to_series(y_sup, total_length, window_size, horizon, step, agg="mean"):
    """
    将监督学习格式 (N, H, F) 还原为原始时间序列长度 (T, F)

    Parameters:
    ----------
    y_sup : np.ndarray
        形状 (N, H, F) 的预测或真实值
    total_length : int
        原始时间序列长度 T
    window_size : int
        历史窗口长度
    horizon : int
        预测步长 H
    step : int
        滑动步长
    agg : str
        聚合方式: "mean" 或 "last"

    Returns:
    -------
    np.ndarray
        形状 (T, F) 的时间序列
    """
    if y_sup.size == 0:
        return np.empty((total_length, 0))

    n_samples, h_len, n_features = y_sup.shape
    if h_len != horizon:
        horizon = h_len

    acc = np.zeros((total_length, n_features), dtype=float)
    cnt = np.zeros((total_length, 1), dtype=float)

    for i in range(n_samples):
        start = i * step + window_size
        end = min(start + horizon, total_length)
        cur_len = end - start
        if cur_len <= 0:
            continue
        acc[start:end] += y_sup[i, :cur_len]
        cnt[start:end] += 1

    if agg == "mean":
        return acc / np.maximum(cnt, 1.0)
    if agg == "last":
        out = np.zeros((total_length, n_features), dtype=float)
        for i in range(n_samples):
            start = i * step + window_size
            end = min(start + horizon, total_length)
            cur_len = end - start
            if cur_len <= 0:
                continue
            out[start:end] = y_sup[i, :cur_len]
        return out

    raise ValueError("agg must be 'mean' or 'last'")

def auto_select_k_for_evaluation(X, k_range=(2, 10)):
    """为评估自动选择最佳聚类数"""
    X_no_nan = np.nan_to_num(X, nan=0.0, copy=True)
    
    if len(X_no_nan) < 2:
        return 2

    # 确保 X 是 3D
    if X_no_nan.ndim != 3:
        return 2

    n_samples, n_channels, n_timesteps = X_no_nan.shape

    # 如果时间步太少，MiniRocket 会报错（需 >=9）
    if n_timesteps < 9:
        pad_width = ((0, 0), (0, 0), (0, 9 - n_timesteps))
        X_padded = np.pad(X_no_nan, pad_width, mode='constant', constant_values=0.0)
    else:
        X_padded = X_no_nan

    try:
        # 使用 MiniRocket 提取特征
        minirocket_transformer = MiniRocket(random_state=42)
        minirocket_transformer.fit(X_padded)
        X_features = minirocket_transformer.transform(X_padded)

        # PCA 降维
        pca = PCA(n_components=min(100, X_features.shape[0], X_features.shape[1]), random_state=42)
        X_features = pca.fit_transform(X_features)

        best_k = k_range[0]
        best_score = -100.0

        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(X_features):
                break

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(X_features)

            if len(np.unique(labels)) < 2:
                continue

            try:
                sil = silhouette_score(X_features, labels, sample_size=min(5000, len(X_features)))
                ch = calinski_harabasz_score(X_features, labels)
                score = 0.6 * sil + 0.4 * (ch / (ch + 1e-8))
            except Exception:
                score = -100.0

            if score > best_score:
                best_score = score
                best_k = k

        return best_k
    except Exception as e:
        print(f"Auto select k error: {e}")
        return 2


def evaluate_downstream_task_performance(data, label, task_type, model_type='proxy', train_ratio=0.6, numeric_cols=None):
    """
    评估修复后数据在下游任务上的性能
    
    Parameters:
    ----------
    dirty_data : np.ndarray
        修复前的数据
    clean_data : np.ndarray
        修复后的数据
    label : np.ndarray
        标签数据（对于分类任务）
    task_type : str
        任务类型 ('classification', 'forecast', 'clustering')
    train_ratio : float
        训练集比例
    
    Returns:
    -------
    dict
        包含各种评估指标的字典，包含代理模型和深度学习模型的结果
    """
    if data.size == 0 or data.shape[0] == 0:
        return {"proxy_performance": 0.0, "final_performance": 0.0}
    
    # 数据划分
    split_idx = int(train_ratio * data.shape[1])
    X_train_temp = data[:, :split_idx, numeric_cols]  
    X_test_temp = data[:, split_idx:, numeric_cols]
    y_train_temp = None
    y_test_temp = None


    if len(X_train_temp) < 1:
        return {"performance": 0.0}

    try:
       return _evaluate_forecast_task(X_train_temp, X_test_temp, model_type)
    except Exception as e:
        print(f"Downstream task evaluation error: {e}")
        return {"performance": 0.0}




def _evaluate_forecast_task(X_train, X_test, model_type='proxy'):
    """评估预测任务性能
    
    Parameters:
    ----------
    X_train : np.ndarray
        训练数据
    X_test : np.ndarray
        测试数据
    model_type : str
        模型类型 ('proxy' for proxy model, 'final' for deep learning model)
    
    Returns:
    -------
    dict
        包含性能指标的字典
    """
    X_sup, y_sup = to_supervised(X_train, window_size=window_size_f, horizon=horizon_f)
    if X_sup.size == 0:
        return {"performance": 0.0, "nrmse": 1.0, "corr": 0.0}
    
    X_sup = np.nan_to_num(X_sup)
    y_sup = np.nan_to_num(y_sup)
    
    try:
        if model_type == 'proxy':
            # 使用DLinear作为代理模型
            model = DLinear(seq_len=window_size_f, pred_len=horizon_f, num_features=X_train.shape[-1], batch_size=32)
        else:
            # 使用LSTMForecast作为深度学习模型
            model = LSTMForecast(input_dim=X_train.shape[-1], hidden_dim=64, pred_len=horizon_f, n_epochs=50, batch_size=32)
        
        model.fit(X_sup, y_sup)
        X_test_sup, y_test_sup = to_supervised(X_test, window_size=window_size_f, horizon=horizon_f)
        if X_test_sup.size == 0:
            return {"performance": 0.0, "nrmse": 1.0, "corr": 0.0}
        
        X_test_sup = np.nan_to_num(X_test_sup)
        y_test_sup = np.nan_to_num(y_test_sup)
        y_pred = model.predict(X_test_sup)

        # 将 (N, H, F) 还原为 (T, F)，便于对比/画图
        total_len = X_test.shape[1]
        y_test_2d = supervised_to_series(
            y_test_sup,
            total_len,
            window_size_f,
            horizon_f,
            window_step,
            agg="mean"
        )
        y_pred_2d = supervised_to_series(
            y_pred,
            total_len,
            window_size_f,
            horizon_f,
            window_step,
            agg="mean"
        )
        y_test_3d = y_test_2d[None, ...]
        y_pred_3d = y_pred_2d[None, ...]
        
        # 计算NRMSE
        rmse = np.sqrt(np.mean((y_test_sup - y_pred) ** 2))
        std_val = np.std(y_test_sup)
        nrmse = rmse / (std_val + 1e-8)
        score_nrmse = np.exp(-nrmse)
        
        # 计算相关系数
        pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        true_flat = y_test_sup.reshape(y_test_sup.shape[0], -1)
        
        vp = pred_flat - pred_flat.mean(axis=1, keepdims=True)
        vt = true_flat - true_flat.mean(axis=1, keepdims=True)
        denom = np.sqrt((vp**2).sum(axis=1) * (vt**2).sum(axis=1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corrs = (vp * vt).sum(axis=1) / (denom + 1e-8)
        corrs = np.nan_to_num(corrs)
        avg_corr = np.mean(corrs)
        score_corr = (avg_corr + 1) / 2
        
        # 综合评分
        performance = float(np.clip(0.6 * score_nrmse + 0.4 * score_corr, 0.0, 1.0))
        
        return {
            "performance": performance,
            "nrmse": nrmse,
            "corr": avg_corr,
            "y_test_2d": y_test_2d,
            "y_pred_2d": y_pred_2d,
            "y_test_3d": y_test_3d,
            "y_pred_3d": y_pred_3d
        }
    except Exception as e:
        print(f"Forecast {model_type} model evaluation error: {e}")
        return {"performance": 0.0, "nrmse": 1.0, "corr": 0.0}




def downstream_task(clean_data, dirty_data, label_2_train, task_type, numeric_cols=None, model_type='final', save_dir="/home/yyy/TSC/TSClean/AutoClean/draw_pictures/downstream_results"):
    if model_type == 'proxy':
        downstream_performance_proxy_dirty = evaluate_downstream_task_performance(dirty_data, label_2_train, task_type, model_type='proxy', train_ratio=0.7, numeric_cols=numeric_cols)
        proxy_pred_2d_dirty = downstream_performance_proxy_dirty.get("y_pred_2d")
        proxy_test_2d_dirty = downstream_performance_proxy_dirty.get("y_test_2d")
        
        downstream_performance_proxy_clean = evaluate_downstream_task_performance(clean_data, label_2_train, task_type, model_type='proxy', train_ratio=0.7, numeric_cols=numeric_cols)
        proxy_pred_2d_clean = downstream_performance_proxy_clean.get("y_pred_2d")
        proxy_test_2d_clean = downstream_performance_proxy_clean.get("y_test_2d")
        
        if proxy_pred_2d_dirty is not None and proxy_test_2d_dirty is not None and proxy_pred_2d_clean is not None and proxy_test_2d_clean is not None:
            print(f"代理模型 - y_pred_2d_dirty shape: {proxy_pred_2d_dirty.shape}, y_test_2d_dirty shape: {proxy_test_2d_dirty.shape}")
            print(f"代理模型 - y_pred_2d_clean shape: {proxy_pred_2d_clean.shape}, y_test_2d_clean shape: {proxy_test_2d_clean.shape}")
    else:
        downstream_performance_final_dirty = evaluate_downstream_task_performance(dirty_data, label_2_train, task_type, model_type='final', train_ratio=0.7, numeric_cols=numeric_cols)
        final_pred_2d_dirty = downstream_performance_final_dirty.get("y_pred_2d")
        final_test_2d_dirty = downstream_performance_final_dirty.get("y_test_2d")
        
        downstream_performance_final_clean = evaluate_downstream_task_performance(clean_data, label_2_train, task_type, model_type='final', train_ratio=0.7, numeric_cols=numeric_cols)
        final_pred_2d_clean = downstream_performance_final_clean.get("y_pred_2d")
        final_test_2d_clean = downstream_performance_final_clean.get("y_test_2d")
        
        if final_pred_2d_dirty is not None and final_test_2d_dirty is not None and final_pred_2d_clean is not None and final_test_2d_clean is not None:
            print(f"深度学习模型 - y_pred_2d_dirty shape: {final_pred_2d_dirty.shape}, y_test_2d_dirty shape: {final_test_2d_dirty.shape}")
            print(f"深度学习模型 - y_pred_2d_clean shape: {final_pred_2d_clean.shape}, y_test_2d_clean shape: {final_test_2d_clean.shape}")

    # 保存结果到文件，便于画图或复用
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def save_csv(array_2d, file_name):
            if array_2d is None:
                return
            file_path = os.path.join(save_dir, file_name)
            pd.DataFrame(array_2d).to_csv(file_path, index=False)

        if model_type == 'proxy':
            save_csv(proxy_pred_2d_clean, "proxy_y_pred_2d_clean.csv")
            save_csv(proxy_test_2d_clean, "proxy_y_test_2d_clean.csv")
            save_csv(proxy_pred_2d_dirty, "proxy_y_pred_2d_dirty.csv")
            save_csv(proxy_test_2d_dirty, "proxy_y_test_2d_dirty.csv")
        if model_type == 'final':
            save_csv(final_pred_2d_clean, "final_y_pred_2d_clean.csv")
            save_csv(final_test_2d_clean, "final_y_test_2d_clean.csv")
            save_csv(final_pred_2d_dirty, "final_y_pred_2d_dirty.csv")
            save_csv(final_test_2d_dirty, "final_y_test_2d_dirty.csv")

if __name__ == "__main__":
    
	# 获取原始数据
    # type = 'forecast'
    # dataset_name = 'ETTh1'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
    
    type = 'forecast'
    dataset_name = 'IDF_OilTemp'
    data = load_original_data(dataset_name, 'clean', rate=1)
    label_2_train = None


    if type == 'forecast':
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        print(data.shape)
        dm = DataManager(data, task_type=type)
    else:
        print(data.shape, label_2_train.shape)
        dm = DataManager(data, label_2_train, task_type=type)


    # 保存原始干净数据
    clean_data = dm.clean_data_raw.copy()

    # 注入错误
    dm.inject_errors(
        0.7,
        ["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
        covered_attrs=range(data.shape[-1]),
    )

    dirty_data, error_mask = dm.get_dirty_data_restored()
    
    numeric_cols = list(range(1,dirty_data.shape[-1]))  # 对于numpy数组，找出特征列
    
    downstream_task(clean_data,dirty_data, label_2_train, type, numeric_cols, model_type='proxy', save_dir="/home/yyy/TSC/TSClean/AutoClean/draw_pictures/downstream_results")
