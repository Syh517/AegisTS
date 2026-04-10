import pandas as pd
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    roc_auc_score, f1_score, silhouette_score, 
    calinski_harabasz_score, davies_bouldin_score, # 新增 DBI
    mean_squared_error, accuracy_score, average_precision_score
)
import pickle
import os


import random
import time
import logging
import warnings

from Error_Detection.ts_column import clean_timestamp_column_with_regression

warnings.filterwarnings('ignore')


from Error_Detection.Detector import Detector
from Error_Injection.injector import DataManager
from Datasets.load_dataset import load_single_dataset, sample_data_by_rate, convert_to_unix_timestamp
from Error_Cleaner.tools.missing import call_imputer, missing_methods
from Error_Cleaner.tools.anomaly import call_outlier_modifier, anomaly_methods
from Error_Cleaner.tools.constraints import call_constraints_handler, constraint_methods

# from Error_Cleaner.evaluate import evaluate_cleaning_effectiveness
from Error_Cleaner.EvaluationMetrics import evaluate_cleaning_effectiveness

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.deep_learning import InceptionTimeClassifier

from aeon.clustering.feature_based import Catch22Clusterer
from aeon.clustering.deep_learning import AEDCNNClusterer

from Error_Cleaner.DLiner import DLinear
from Error_Cleaner.LSTMForecast import LSTMForecast


from Error_Cleaner.strategy import (
    save_trained_agents,
    load_trained_agents,
    apply_cleaning_from_saved_model
)

# ==============================
# 常量
# ==============================
LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4 = 0, 0.05, 0.7, 0.25  # -LAMBDA_2 * R_Cost_N + LAMBDA_3 * R_Issue_N + LAMBDA_4 * low_reward_N
LAMBDA_GRAD = 10.0  # 深度学习模型梯度权重
ALPHA = 0.1 # k
MU_1, MU_2, MU_3 = 0.2, 0.2, 0.6 
# 将MAX_COST_NORM从100.0调整为300.0，以适应更广泛的时间成本范围
# 这个值应该根据实际运行时间和任务复杂度进行调整
MAX_COST_NORM = 300.0
K_MIN, K_MAX = 2, 30
K_STABILITY_WINDOW = 3
AEDCNN_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
AEDCNN_BEST_FILE = "aedcnn_best"

HIGH_ACTIONS = ['MISSING', 'ANOMALY', 'CONSTRAINT', 'FINISH']
LOW_ACTIONS_MAP = {
    'MISSING': missing_methods,
    'ANOMALY': anomaly_methods,
    'CONSTRAINT': constraint_methods,
    'FINISH': ['NONE']
}
ISSUE_MAP = {k: i for i, k in enumerate(HIGH_ACTIONS[:-1])}

window_size_f = 30
horizon_f = 10
window_step = 5
hidden_dim_f = 64
# GLOBAL_RANDOM_SEED = 42  # 移除固定随机种子
# np.random.seed(42)  # 设置全局种子

_minirocket_transformer = None


def _sanitize_numeric_array(array, clip_value=1e6, dtype=np.float32):
    """将数组稳定化为有限数值，避免深度模型中间层出现 NaN/Inf。"""
    sanitized = np.asarray(array, dtype=dtype).copy()
    np.nan_to_num(sanitized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(sanitized, -clip_value, clip_value, out=sanitized)
    if not np.isfinite(sanitized).all():
        sanitized[~np.isfinite(sanitized)] = 0.0
    return sanitized


def _to_aeon_ts_input(array):
    """转换为 aeon 期望格式 (n_samples, n_channels, n_timepoints)。"""
    return _sanitize_numeric_array(array).swapaxes(1, 2)


# ==============================
# 工具函数
# ==============================
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


def _build_aedcnn_clusterer(estimator, n_epochs=50, random_state=42, batch_size=32, file_path=None, file_name=AEDCNN_BEST_FILE):
    if file_path is None:
        file_path = AEDCNN_MODEL_DIR
    os.makedirs(file_path, exist_ok=True)
    if not file_path.endswith(os.sep):
        file_path = file_path + os.sep

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(file_path, file_name + ".keras"),
            monitor="loss",
            save_best_only=False,
        )
    ]

    return AEDCNNClusterer(
        estimator=estimator,
        n_epochs=n_epochs,
        random_state=random_state,
        batch_size=batch_size,
        file_path=file_path,
        save_best_model=True,
        best_file_name=file_name,
        callbacks=callbacks,
    )

def _has_consecutive_repeated_actions_with_minimal_gain(env, threshold=0.0003, consecutive_count=2):
    """
    检查是否有连续重复的动作且指标改进微小

    Parameters:
        env: RLCleanEnvironment 实例
        threshold: 指标改进阈值，低于此值视为无改进
        consecutive_count: 需要检查的连续动作次数，默认为2

    Returns:
        bool: 是否存在连续重复动作且指标改进微小
        tuple: 重复的动作 (high_action, low_action) 或 None
    """
    if not hasattr(env, 'action_history') or not hasattr(env, 'performance_history'):
        return False, None
    
    if len(env.action_history) < consecutive_count:
        return False, None
    
    # 检查最近的consecutive_count次动作
    recent_actions = env.action_history[-consecutive_count:]  # 最近consecutive_count次动作
    recent_performances = env.performance_history[-consecutive_count:] if len(env.performance_history) >= consecutive_count else []
    
    # 如果最近consecutive_count次动作相同且指标改进都小于阈值
    if (len(recent_actions) == consecutive_count and 
        all(action == recent_actions[0] for action in recent_actions) and 
        len(recent_performances) == consecutive_count):
        # 检查这consecutive_count次连续重复动作的指标变化量是否都很小
        if all(abs(perform) <= threshold for perform in recent_performances):
            return True, recent_actions[0]
    
    return False, None

def auto_select_k(X, k_range=(2, 10)):
    """
    基于 MiniRocket 特征自动选择最优聚类数 k。
    
    Parameters:
        X: numpy array of shape (n_samples, n_channels, n_timestamps)
        k_range: tuple (min_k, max_k)
    
    Returns:
        best_k: int
        best_score: float
    """
    global _minirocket_transformer

    X = _sanitize_numeric_array(X)  # 确保没有 NaN/Inf 且避免极值溢出
    
    if len(X) < 10:
        return 2, -100.0

    # 确保 X 是 3D
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input (N, C, T), got shape {X.shape}")

    n_samples, n_channels, n_timesteps = X.shape

    # 如果时间步太少，MiniRocket 会报错（需 >=9）
    if n_timesteps < 9:
        # 零填充到 9（或使用更小的 kernel？但 MiniRocket 固定最小 9）
        pad_width = ((0, 0), (0, 0), (0, 9 - n_timesteps))
        X_padded = np.pad(X, pad_width, mode='constant', constant_values=0.0)
    else:
        X_padded = X

    # 使用 MiniRocket 提取特征（保留时间结构！）
    if _minirocket_transformer is None:
        _minirocket_transformer = MiniRocket(random_state=42)
        _minirocket_transformer.fit(X_padded)  # fit once on first call
    else:
        # 注意：transformer 已 fit，直接 transform
        pass

    # 提取特征：(n_samples, ~10,000)
    X_features = _minirocket_transformer.transform(X_padded)
    X_features = _sanitize_numeric_array(X_features)

    # 可选：PCA 降维（缓解高维稀疏性，加速聚类）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(100, X_features.shape[0], X_features.shape[1]), random_state=42)
    X_features = pca.fit_transform(X_features)

    best_k = k_range[0]
    best_score = -100.0

    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(X_features):
            break  # 聚类数不能 >= 样本数

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(50, len(X_features) // 3),
            random_state=42,
            max_iter=100,
            n_init=3  # 减少初始化次数以加速
        )
        labels = kmeans.fit_predict(X_features)

        if len(np.unique(labels)) < 2:
            continue

        try:
            sil = silhouette_score(X_features, labels, sample_size=min(5000, len(X_features)))
            ch = calinski_harabasz_corrected(X_features, labels)  # 见下方说明
            score = 0.6 * sil + 0.4 * (ch / (ch + 1e-8))  # 避免除零
        except Exception:
            score = -100.0

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score

def calinski_harabasz_corrected(X, labels):
    """安全版 CH score，避免除零"""
    try:
        return calinski_harabasz_score(X, labels)
    except Exception:
        return 0.0

def calculate_structure_metric(col_data):
    # 输入col_data始终为numpy数组
    if col_data.shape[0] < 2:
        return 0.0
    
    k = min(3, col_data.shape[0])
    kernel = np.ones(k) / k
    rm = np.convolve(col_data, kernel, mode='same')
    return -np.mean(np.abs(col_data - rm))

def to_supervised(data_3d, window_size, horizon):
    series = data_3d.squeeze(0)  # (T, F)
    T, F = series.shape
    
    min_required = window_size + horizon
    
    # 如果长度不足，进行前向填充（Padding）
    if T < min_required:
        pad_len = min_required - T + 1
        # 在序列开头填充0或者序列首部值
        pad_width = ((pad_len, 0), (0, 0))
        series = np.pad(series, pad_width, mode='edge')
        T = series.shape[0]
    
    X, y = [], []
    # 确保至少能循环一次
    for t in range(0, T - window_size - horizon + 1, window_step):
        X_window = series[t : t + window_size]
        Y_future = series[t + window_size : t + window_size + horizon]
        X.append(X_window)
        y.append(Y_future)
    
    return np.array(X), np.array(y)

def to_supervised_evaluation(data_3d, window_size, horizon):
    """
    将单条多变量时间序列 (1, T, F) 转换为监督学习格式 (N, window_size, F)
    使用非重叠窗口，防止评估指标虚高
    """
    series = data_3d.squeeze(0)  # (T, F)
    T, F = series.shape
    
    min_required = window_size + horizon
    
    # 如果长度不足，进行前向填充（Padding）
    if T < min_required:
        pad_len = min_required - T + 1
        # 在序列开头填充0或者序列首部值
        pad_width = ((pad_len, 0), (0, 0))
        series = np.pad(series, pad_width, mode='edge')
        T = series.shape[0]
    
    X, y = [], []
    # 使用非重叠窗口，防止评估指标虚高
    for t in range(0, T - window_size - horizon + 1, horizon):  # 使用horizon作为步长，确保窗口不重叠
        X_window = series[t : t + window_size]
        Y_future = series[t + window_size : t + window_size + horizon]
        X.append(X_window)
        y.append(Y_future)
    
    return np.array(X), np.array(y)

# ==============================
# 统一训练+评估
# ==============================
def train_and_evaluate(task_type, model_type, model_instance, env):
    """
    训练并评估模型性能。
    - Proxy 模型：返回 (性能变化, 训练时间)
    - Final 模型：返回 (绝对性能, 训练时间)
    """
    # 使用局部随机状态对象，而不是全局设置随机种子
    local_np_random = np.random.RandomState(42)

    if env.cur_train.size == 0 or env.cur_train.shape[0] == 0:
        return (-100.0, 1.0) if model_type == 'final' else (0.0, 1.0)

    if env.task_type =='forecast':
        # 对于预测任务，按时间维度分割训练测试集（每个样本分别分割）
        split_idx = int(env.train_ratio * env.cur_train.shape[1])
        X_train_temp = env.cur_train[:, :split_idx, env.numeric_cols]  
        X_test_temp = env.cur_train[:, split_idx:, env.numeric_cols]
        y_train_temp = None
        y_test_temp = None
    else:
        # 对于分类和聚类任务，使用随机抽样而非连续切片
        n_samples = env.cur_train.shape[0]
        indices = np.arange(n_samples)
        
        # 随机选择训练集索引
        train_size = int(env.train_ratio * n_samples)
        
        if env.train_indices is None or env.test_indices is None:
            if env.cur_label is not None:
                # 确保训练集和测试集都包含所有标签类别，且每类至少有2个样本在训练集中
                unique_labels = np.unique(env.cur_label)
                
                # 为每个唯一标签选择样本放入训练集和测试集
                train_indices = []
                test_indices = []
                remaining_indices = indices.copy()
                
                for label in unique_labels:
                    # 获取当前标签的所有索引
                    label_indices = np.where(env.cur_label == label)[0]
                    available_indices = np.intersect1d(label_indices, remaining_indices)
                    
                    if len(available_indices) >= 2:
                        # 至少为当前标签选择2个样本放入训练集
                        selected_train_idx = local_np_random.choice(available_indices, size=2, replace=False)
                        train_indices.extend(selected_train_idx)
                        remaining_indices = np.setdiff1d(remaining_indices, selected_train_idx)
                        
                        # 如果还有剩余的同类样本，可以选择部分放入测试集
                        remaining_label_indices = np.setdiff1d(available_indices, selected_train_idx)
                        if len(remaining_label_indices) > 0:
                            # 确保测试集也有代表性，至少保留一个样本在测试集（如果可能）
                            if len(remaining_label_indices) == 1:
                                # 只剩一个样本，直接加入测试集
                                test_indices.extend(remaining_label_indices)
                                remaining_indices = np.setdiff1d(remaining_indices, remaining_label_indices)
                            else:
                                # 多余样本，按比例分配到测试集
                                test_size_for_label = max(1, int(len(remaining_label_indices) * (1-env.train_ratio)))
                                test_size_for_label = min(test_size_for_label, len(remaining_label_indices))
                                selected_test_idx = local_np_random.choice(remaining_label_indices, size=test_size_for_label, replace=False)
                                test_indices.extend(selected_test_idx)
                                remaining_indices = np.setdiff1d(remaining_indices, selected_test_idx)
                    elif len(available_indices) == 1:
                        # 只有一个样本，优先放入训练集
                        train_indices.extend(available_indices)
                        remaining_indices = np.setdiff1d(remaining_indices, available_indices)
                
                # 从剩余的索引中继续选择样本直到达到训练集大小（如果需要）
                additional_needed = max(0, train_size - len(train_indices))
                if additional_needed > 0 and len(remaining_indices) > 0:
                    additional_indices = local_np_random.choice(
                        remaining_indices, 
                        size=min(additional_needed, len(remaining_indices)), 
                        replace=False
                    )
                    train_indices.extend(additional_indices)
                    remaining_indices = np.setdiff1d(remaining_indices, additional_indices)
                
                # 将剩余的索引全部加入测试集
                test_indices.extend(remaining_indices)
                
                train_indices = np.array(train_indices)
                test_indices = np.array(test_indices)
                
            else:
                # 对于非分类任务或没有标签的情况，使用完全随机抽样
                train_indices = local_np_random.choice(indices, size=train_size, replace=False)
                test_indices = np.setdiff1d(indices, train_indices)  # 剩余索引作为测试集

            env.train_indices = train_indices
            env.test_indices = test_indices
        else:
            train_indices = env.train_indices
            test_indices = env.test_indices

        

        X_train_temp = env.cur_train[train_indices][:, :, env.numeric_cols].copy()
        X_test_temp = env.cur_train[test_indices][:, :, env.numeric_cols].copy()
        
        if env.task_type == 'classification' and env.cur_label is not None:
            y_train_temp = env.cur_label[train_indices]
            y_test_temp = env.cur_label[test_indices]
        else:
            y_train_temp = None
            y_test_temp = None


    if len(X_train_temp) < 1:
        return (-100.0, 1.0) if model_type == 'final' else (0.0, 1.0)

    start_time = time.time()

    # ======================
    # PROXY MODEL 
    # ======================
    if model_type == 'proxy':
        last_test_perf = getattr(env, 'last_test_perf', 0.0)
        print("last_test_perf",last_test_perf)
        try:
            if task_type == 'classification':
                 # 转置数据以匹配模型期望的格式 (n_samples, n_channels, n_timepoints)
                X = _to_aeon_ts_input(X_train_temp)
                y = y_train_temp
                if len(np.unique(y)) < 2:
                    perf_change = 0.0
                else:
                    classes = np.unique(y)
                    model_instance.fit(X, y)
                    cur_test_perf = _evaluate_on_test(model_instance, env, X_test_temp, y_test_temp)
                    perf_change = cur_test_perf - last_test_perf

            elif task_type == 'forecast':
                # 对每个样本分别进行训练

                # 对每个样本分别生成监督学习数据，然后合并
                all_X_sup = []
                all_y_sup = []
                for sample_idx in range(X_train_temp.shape[0]):
                    X_single_sample = X_train_temp[sample_idx:sample_idx+1]  # (1, T, F)
                    X_sup, y_sup = to_supervised(X_single_sample, window_size=window_size_f, horizon=horizon_f)
                    if X_sup.size != 0:
                        X_sup = np.nan_to_num(X_sup)
                        y_sup = np.nan_to_num(y_sup)

                        all_X_sup.append(X_sup)
                        all_y_sup.append(y_sup)
                
                # 合并所有样本的数据
                if all_X_sup:
                    X_sup_combined = np.concatenate(all_X_sup, axis=0)
                    y_sup_combined = np.concatenate(all_y_sup, axis=0)
                    X_sup_combined = np.nan_to_num(X_sup_combined)
                    y_sup_combined = np.nan_to_num(y_sup_combined)

                    # 直接使用当前数据进行训练，不累积历史数据
                    model_instance.fit(X_sup_combined, y_sup_combined)
                
                # 评估整体性能变化
                cur_test_perf = _evaluate_on_test(model_instance, env, X_test_temp, y_test_temp)
                perf_change = cur_test_perf - last_test_perf

            elif task_type == 'clustering':
                X_all = env.cur_train[:, :, env.numeric_cols].copy()
                X_all = _to_aeon_ts_input(X_all)  # 转置为 (n_samples, n_channels, n_timepoints)
                k, _ = auto_select_k(X_all)
                env.best_k = k  # 更新 best_k
                model_instance.estimator.set_params(n_clusters=k)


                # 转置数据以匹配模型期望的格式 (n_samples, n_channels, n_timepoints)
                X = _to_aeon_ts_input(X_train_temp)
                model_instance.fit(X)
                cur_test_perf = _evaluate_on_test(model_instance, env, X_test_temp, y_test_temp)
                perf_change = cur_test_perf - last_test_perf

        except Exception as e:
            print(f"Proxy training error: {e}")
            perf_change = 0.0

        training_time = time.time() - start_time + 0.001
        return perf_change, training_time

    # ======================
    # FINAL MODEL
    # ======================
    else:
        # try:
        if task_type == 'classification':
            if hasattr(model_instance, '_model_params'):
                model = InceptionTimeClassifier(**model_instance._model_params)
            else:
                model = InceptionTimeClassifier(n_epochs=150, random_state=42, batch_size=32)
            X_train = _to_aeon_ts_input(X_train_temp)
            y_train = y_train_temp
            if len(np.unique(y_train)) < 2:
                final_perf = 0.0
            else:
                classes = np.unique(y_train)
                model.fit(X_train, y_train)
                final_perf = _evaluate_on_test(model, env, X_test_temp, y_test_temp)

        elif task_type == 'forecast':
            if hasattr(model_instance, '_model_params'):
                model = LSTMForecast(**model_instance._model_params)
            else:
                model = LSTMForecast(input_dim=env.cur_train[:, :, env.numeric_cols].shape[-1], hidden_dim=hidden_dim_f, pred_len=horizon_f, n_epochs=50, batch_size=32)

            # 对每个样本分别生成监督学习数据，然后合并
            all_X_sup = []
            all_y_sup = []
            for sample_idx in range(X_train_temp.shape[0]):
                X_single_sample = X_train_temp[sample_idx:sample_idx+1]  # (1, T, F)
                X_sup, y_sup = to_supervised(X_single_sample, window_size=window_size_f, horizon=horizon_f)
                if X_sup.size != 0:
                    X_sup = np.nan_to_num(X_sup)
                    y_sup = np.nan_to_num(y_sup)

                    all_X_sup.append(X_sup)
                    all_y_sup.append(y_sup)
            
            # 合并所有样本的数据
            if all_X_sup:
                X_sup_combined = np.concatenate(all_X_sup, axis=0)
                y_sup_combined = np.concatenate(all_y_sup, axis=0)
                X_sup_combined = np.nan_to_num(X_sup_combined)
                y_sup_combined = np.nan_to_num(y_sup_combined)
                # 直接使用当前数据进行训练，不累积历史数据
                model.fit(X_sup_combined, y_sup_combined)
            # 如果没有有效的训练数据，跳过训练步骤
                
            # 评估整体性能变化
            final_perf = _evaluate_on_test(model, env, X_test_temp, y_test_temp)

        elif task_type == 'clustering':
            X_all = env.cur_train[:, :, env.numeric_cols].copy()
            X_all = _to_aeon_ts_input(X_all)  # 转置为 (n_samples, n_channels, n_timepoints)
            k, _ = auto_select_k(X_all)
            env.best_k = k  # 更新 best_k
            if hasattr(model_instance, '_model_params'):
                base_params = model_instance._model_params.copy()
                estimator = base_params.get('estimator', KMeans())
                estimator.set_params(n_clusters=k)
                base_params['estimator'] = estimator
                model = _build_aedcnn_clusterer(
                    estimator=estimator,
                    n_epochs=base_params.get('n_epochs', 50),
                    random_state=base_params.get('random_state', 42),
                    batch_size=base_params.get('batch_size', 32),
                    file_path=base_params.get('file_path', AEDCNN_MODEL_DIR),
                    file_name=base_params.get('file_name', AEDCNN_BEST_FILE),
                )
            else:
                model = _build_aedcnn_clusterer(
                    estimator=KMeans(n_clusters=k, random_state=42),
                    n_epochs=50,
                    random_state=42,
                    batch_size=32,
                    file_path=AEDCNN_MODEL_DIR,
                    file_name=AEDCNN_BEST_FILE,
                )

            # 转置数据以匹配模型期望的格式 (n_samples, n_channels, n_timepoints)
            X_train = _to_aeon_ts_input(X_train_temp)
            model.fit(X_train)
            final_perf = _evaluate_on_test(model, env, X_test_temp, y_test_temp)

        # except Exception as e:
        #     print(f"Final model training error: {e}")
        #     final_perf = 0.0

        training_time = time.time() - start_time + 0.001
        return final_perf, training_time


def _evaluate_on_test(model, env, X_test_temp, y_test_temp):
    """
    改进后的统一评估函数
    1. Forecast: 增加相关系数(Corr)评估，使用Std归一化NRMSE
    2. Clustering: 展平时间维度以保留形状信息，引入DBI替代不稳定的CH归一化
    3. Classification: 使用Macro F1以应对不平衡
    """
    if env.cur_train.size == 0:
        return 0.0

    try:
        # ===========================
        # 1. Classification
        # ===========================
        if env.task_type == 'classification':
            # (N, C, T)
            X_test = _to_aeon_ts_input(X_test_temp)
            
            y_pred = model.predict(X_test)
            
            # 修改点：使用 macro 平均以更好地处理类别不平衡
            f1 = f1_score(y_test_temp, y_pred, average='macro')
            
            y_score = model.predict_proba(X_test)        # (n, classes)
            y_true = y_test_temp                         # (n,)，可能只含部分标签

            auc = roc_auc_score(
                y_true,
                y_score,
                multi_class='ovr',          # 必须指定
                average='weighted',         # 或 'macro'
                labels=model.classes_       # 强制对齐列顺序（强烈建议）
            )

                
            # 返回F1和AUC的平均值作为综合指标
            return float((f1 + auc) / 2)


        # ===========================
        # 2. Forecast
        # ===========================
        elif env.task_type == 'forecast':
            # 为每个样本单独处理预测
            if X_test_temp.shape[0] == 0:
                return 0.0

            # 收集所有样本的评估结果
            all_scores = []
            
            for sample_idx in range(X_test_temp.shape[0]):
                X_single_sample = X_test_temp[sample_idx:sample_idx+1]  # 保持3D格式 (1, T, F)
                
                # 为单个样本生成监督数据
                X_test_sup, y_test_sup = to_supervised_evaluation(X_single_sample, window_size=window_size_f, horizon=horizon_f)
                
                if X_test_sup.size == 0:
                    continue

                X_test_sup = np.nan_to_num(X_test_sup)
                y_test_sup = np.nan_to_num(y_test_sup)

                eval_model = model
                y_pred = eval_model.predict(X_test_sup)

                # --- 修改点开始 ---
                # 1. NRMSE (使用标准差归一化，防止极值使得分母过大导致误差被低估)
                rmse = np.sqrt(np.mean((y_test_sup - y_pred) ** 2))

                # 优化分母逻辑：取 极差 和 标准差 中较大的一个，并加入 epsilon 防止除零
                denom = np.max(y_test_sup) - np.min(y_test_sup)
                if denom < 1e-6: 
                    # 如果极差太小（说明是常数序列），改用标准差，若标准差也小，则给一个基准值
                    denom = np.std(y_test_sup) + 1e-6
                nrmse = rmse / denom

                # std_val = np.std(y_test_sup)
                # nrmse = rmse / (std_val + 1e-8)
                
                # 2. CORR (形状相关性) - 奖励能够拟合趋势的模型
                # 计算每个样本的相关系数并取平均
                # y_pred, y_test_sup shape: (N, H, F) -> flatten to (N, H*F) for correlation
                pred_flat = y_pred.reshape(y_pred.shape[0], -1)
                true_flat = y_test_sup.reshape(y_test_sup.shape[0], -1)
                
                # 简单的批量相关系数计算
                vp = pred_flat - pred_flat.mean(axis=1, keepdims=True)
                vt = true_flat - true_flat.mean(axis=1, keepdims=True)
                denom = np.sqrt((vp**2).sum(axis=1) * (vt**2).sum(axis=1))
                
                # 处理常数序列导致的除零 (denom=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    corrs = (vp * vt).sum(axis=1) / (denom + 1e-8)
                corrs = np.nan_to_num(corrs) # 将NaN (常数序列) 设为 0
                avg_corr = np.mean(corrs)

                # 3. 组合分数
                # NRMSE 越小越好 (限制在 0~1 之间反转), Corr 越大越好 (-1~1)
                # 使用指数衰减处理 NRMSE
                tau = 1.0  # 可调参数
                score_nrmse = np.exp(-nrmse / tau)
                score_corr = (avg_corr + 1) / 2 # 归一化到 0~1

                # 综合评分: 60% 准确度, 40% 趋势
                score = 0.6 * score_nrmse + 0.4 * score_corr
                # --- 修改点结束 ---
                
                all_scores.append(float(np.clip(score, 0.0, 1.0)))
            
            # 如果没有可评估的样本，返回0
            if not all_scores:
                return 0.0
                
            # 返回所有样本的平均分数
            return float(np.mean(all_scores))


        # ===========================
        # 3. Clustering
        # ===========================
        elif env.task_type == 'clustering':
            # (N, C, T)
            X_test = _to_aeon_ts_input(X_test_temp)
            labels = model.predict(X_test)
            
            if len(np.unique(labels)) < 2:
                return 0.0
            
            # --- 修改点开始 ---
            # 1. 数据重塑：保留时间维度特征
            # 不要使用 mean(axis=2)，而是展平 (N, C*T) 以保留形状信息
            n_samples, n_channels, n_timesteps = X_test.shape
            X_test_flat = X_test.reshape(n_samples, -1)
            
            # 如果维度过高，Silhoutte 计算会很慢，且距离度量在高维下失效
            # 可选：简单的下采样 (每隔3个点取一个) 以加速
            if X_test_flat.shape[1] > 500:
                X_test_flat = X_test_flat[:, ::3]

            # 2. Silhouette Score (范围 -1 到 1)
            # 采样计算以加速 (最多计算 2000 个样本)
            sample_size = min(2000, n_samples)
            sil = silhouette_score(X_test_flat, labels, sample_size=sample_size)
            norm_sil = (sil + 1) / 2  # 归一化到 0~1

            # 3. Davies-Bouldin Index (DBI) 替代 CH Score
            # DBI 越小越好 (最小值为0)，不需要像 CH 那样猜测分母
            dbi = davies_bouldin_score(X_test_flat, labels)
            
            # 将 DBI 转换为 0~1 分数 (DBI 常见范围 0.5 ~ 5.0)
            # 使用 1 / (1 + DBI) 这是一个非常稳定的归一化方法
            norm_dbi = 5.0 / (5.0 + dbi)
            # norm_dbi = np.exp(-dbi)

            score = 0.5 * norm_sil + 0.5 * norm_dbi
            # --- 修改点结束 ---
            
            return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0


# ==============================
# 环境
# ==============================
class RLCleanEnvironment:
    def __init__(self, dirty_data, label, task_type, detector, constraints, concentrated=True, max_steps=20, train_ratio=0.7):
        # 输入数据为dirty_data和label两个独立的numpy数组
        # 验证dirty_data是三维的
        if not isinstance(dirty_data, np.ndarray):
            raise TypeError("dirty_data must be a numpy array")
        if len(dirty_data.shape) != 3:
            raise ValueError(f"dirty_data must be 3-dimensional, but got shape {dirty_data.shape}")
        
        
        # 验证label是一维的
        if label is not None:
            if not isinstance(label, np.ndarray):
                raise TypeError("label must be a numpy array")
            if len(label.shape) != 1:
                raise ValueError(f"label must be 1-dimensional, but got shape {label.shape}")
        
        self.dirty_data = dirty_data.copy()
        self.label = label.copy() if label is not None else None
        self.cur_train = self.dirty_data.copy()
        self.cur_label = self.label.copy() if self.label is not None else None
        self.task_type = task_type
        self.detector = detector
        self.constraints = constraints
        self.max_steps = max_steps
        self.train_ratio = train_ratio
        self.concentrated = concentrated

        self.train_indices = None
        self.test_indices = None

        # 检查cur_train中的缺失值位置，生成missing_mask
        self.missing_mask = np.isnan(self.cur_train)

        self.HIGH_ACTIONS = HIGH_ACTIONS
        self.LOW_ACTIONS_MAP = LOW_ACTIONS_MAP
        self.ISSUE_MAP = ISSUE_MAP

        # 对于numpy数组，找出特征列
        self.numeric_cols = list(range(1,dirty_data.shape[-1]))  # 修改为第三维

        self.step_count = 0
        self.last_macro_action = 'NONE'
        self.total_cost = 0.0
        self.k_history = []
        self.best_k = 2
        # 初始化缓存检测结果
        self.cached_detection_results = None
        
        # 添加重复动作跟踪
        self.action_history = []  # 记录(high_action, low_action)对
        self.performance_history = []  # 记录对应的性能变化
        self.action_count = {}  # 记录每个动作的执行次数
        self.recent_action_effects = {}  # 记录每个动作的近期效果
        self.current_issue_severity = None
        self.prev_issue_severity = None
    

        self.proxy_model = self._create_proxy_model()
        self.final_model = self._create_final_model()

        # 移除不必要的预训练代码，因为train_and_evaluate会在每次评估时重新训练模型
        if self.task_type == 'clustering':
            self._update_best_k()

        # 获取三次初始性能并选择最差的一个
        initial_proxy_perfs = []
        initial_final_perfs = []
        
        for i in range(3):
            proxy_perf, _ = train_and_evaluate(self.task_type, 'proxy', self.proxy_model, self)
            final_perf, _ = train_and_evaluate(self.task_type, 'final', self.final_model, self)
            initial_proxy_perfs.append(proxy_perf)
            initial_final_perfs.append(final_perf)
        
        # 选择最差的初始性能（最小值）
        self.initial_proxy_perf = min(initial_proxy_perfs)
        self.initial_final_perf = min(initial_final_perfs)
        
        self.last_test_perf = self.initial_proxy_perf
        self.prev_test_perf = self.initial_proxy_perf
        initial_rates, _, _, _, _, _ = self._get_issue_rates()
        self.initial_terminal_issue_severity = float(np.sum(initial_rates))
        self.initial_terminal_quality_reward = np.clip(1.0 - self.initial_terminal_issue_severity, -1.0, 1.0)
        print(f"【初始】代理模型性能: {self.initial_proxy_perf:.4f}")
        print(f"【初始】深度学习模型性能: {self.initial_final_perf:.4f}")

    def _create_proxy_model(self):
        if self.task_type == 'classification':
            return MiniRocketClassifier(n_kernels=5000, random_state=42)
        elif self.task_type == 'forecast':
            return DLinear(seq_len=window_size_f, pred_len=horizon_f, num_features=self.cur_train.shape[-1], batch_size=32)    
        elif self.task_type == 'clustering':
            # 先计算基于实际数据的best_k值
            X = self.cur_train[:, :, self.numeric_cols].copy()
            X = _to_aeon_ts_input(X)  # 转置为 (n_samples, n_channels, n_timepoints)
            best_k, _ = auto_select_k(X, (K_MIN, K_MAX))
            self.best_k = best_k  # 更新best_k
            return Catch22Clusterer(estimator=KMeans(n_clusters=self.best_k, random_state=42))
        
    def _create_final_model(self):
        if self.task_type == 'classification':
            model = InceptionTimeClassifier(n_epochs=150, random_state=42, batch_size=32)
            model._is_final_model = True 
            model._model_params = {'n_epochs': 150, 'random_state': 42, 'batch_size': 32}
            return model
        elif self.task_type == 'forecast':
            model = LSTMForecast(input_dim=self.cur_train[:, :, self.numeric_cols].shape[-1], hidden_dim=hidden_dim_f, pred_len=horizon_f, n_epochs=50, batch_size=32)
            model._is_final_model = True
            model._model_params = {
                'input_dim': self.cur_train[:, :, self.numeric_cols].shape[-1], 
                'hidden_dim': hidden_dim_f, 
                'pred_len': horizon_f, 
                'n_epochs': 50, 
                'batch_size': 32
            }
            return model
        elif self.task_type == 'clustering':
            # 先计算基于实际数据的best_k值
            X = self.cur_train[:, :, self.numeric_cols].copy()
            X = _to_aeon_ts_input(X)
            best_k, _ = auto_select_k(X, (K_MIN, K_MAX))
            self.best_k = best_k  # 更新best_k
            
            estimator = KMeans(n_clusters=self.best_k, random_state=42)
            model = _build_aedcnn_clusterer(
                estimator=estimator,
                n_epochs=50,
                random_state=42,
                batch_size=32,
                file_path=AEDCNN_MODEL_DIR,
                file_name=AEDCNN_BEST_FILE,
            )
            model._is_final_model = True
            model._model_params = {
                'estimator': estimator,
                'n_epochs': 50,
                'random_state': 42,
                'batch_size': 32,
                'file_path': AEDCNN_MODEL_DIR,
                'file_name': AEDCNN_BEST_FILE
            }
            return model

    def _skew(self, data):
        """计算偏度"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew

    def _kurtosis(self, data):
        """计算峰度"""
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurt
    def _calculate_overall_structure(self, data):
        """
        计算整个数据集的结构度量
        """
        if not self.numeric_cols or data.size == 0:
            return 0.0
            
        total_structure = []
        
        # 输入数据始终为numpy数组，处理三维数据 (n_samples, n_timesteps, n_features)
        for sample_idx in range(data.shape[0]):
            sample_data = data[sample_idx, :, :]

            structure = []
            for col_idx in self.numeric_cols:
                col_data = sample_data[:, col_idx]
                # 处理缺失值并计算结构度量
                filled_data = np.nan_to_num(col_data)  # 简单处理缺失值
                structure.append(calculate_structure_metric(filled_data))

            sample_avg_smoothness = np.mean(structure) if structure else 0.0
            total_structure.append(sample_avg_smoothness)

        return np.mean(total_structure) if total_structure else 0.0

    def _get_issue_rates(self):
        if not self.numeric_cols:
            # 返回默认值和空的检测结果
            return [0.0, 0.0, 0.0], None, None, None, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        missing_rates, outlier_rates, outlier_predict_labels, violation_rates, _ = self.detector.detect_all(self.cur_train)
        avg_missing_rates = np.mean(missing_rates)
        avg_outlier_rates = np.mean(outlier_rates)
        avg_violation_rates = np.mean([v.get('total', 0) for v in violation_rates.values()])
        
        # 缓存检测结果，供后续使用
        self.cached_detection_results = (
            missing_rates, 
            outlier_rates, 
            outlier_predict_labels, 
            violation_rates
        )
        print("检测指标：",[avg_missing_rates, avg_outlier_rates, avg_violation_rates])

        avg_row_violation_rates = np.mean([v.get('row_constraints', 0) for v in violation_rates.values()])
        avg_speed_violation_rates = np.mean([v.get('speed_constraints', 0) for v in violation_rates.values()])
        avg_accel_violation_rates = np.mean([v.get('accel_constraints', 0) for v in violation_rates.values()])
        avg_variance_violation_rates = np.mean([v.get('variance_constraints', 0) for v in violation_rates.values()])
        avg_od_violation_rates = np.mean([v.get('od_constraints', 0) for v in violation_rates.values()])
        avg_dd_violation_rates = np.mean([v.get('dd_constraints', 0) for v in violation_rates.values()])

        print("约束违反：", avg_row_violation_rates, avg_speed_violation_rates, avg_accel_violation_rates, avg_variance_violation_rates, avg_od_violation_rates, avg_dd_violation_rates)

        # 返回更详细的违规率信息
        return [
            avg_missing_rates, 
            avg_outlier_rates, 
            avg_violation_rates,
        ], missing_rates, outlier_rates, outlier_predict_labels, violation_rates,[
            avg_row_violation_rates,
            avg_speed_violation_rates,
            avg_accel_violation_rates,
            avg_variance_violation_rates,
            avg_od_violation_rates,
            avg_dd_violation_rates]

    def _state_high(self, rates=None):
        if rates is None:
            rates, _, _, _, _, _ = self._get_issue_rates()
        max_rate_idx = np.argmax(rates) if any(rates) else -1
        quality_one_hot = np.zeros(len(self.ISSUE_MAP))
        if max_rate_idx != -1:
            quality_one_hot[max_rate_idx] = 1.0

        cur_issue_severity = float(np.sum(rates))
        self.current_issue_severity = cur_issue_severity

        action_oh = np.zeros(len(self.HIGH_ACTIONS))
        if self.last_macro_action in self.HIGH_ACTIONS:
            action_oh[self.HIGH_ACTIONS.index(self.last_macro_action)] = 1.0

        normalized_cost = min(self.total_cost / MAX_COST_NORM, 1.0)
        # 添加边界检查，确保k_norm在[0, 1]范围内
        k_norm = (self.best_k - K_MIN) / (K_MAX - K_MIN)
        k_norm = np.clip(k_norm, 0.0, 1.0)  # 限制在[0, 1]范围内
        
        # 添加动作历史信息到状态中
        action_history_features = np.zeros(len(self.HIGH_ACTIONS) * 3)  # 最近3个动作
        if self.action_history:
            # 获取最近3次动作
            recent_actions = self.action_history[-3:]
            for i, (high_act, _) in enumerate(recent_actions):
                if high_act in self.HIGH_ACTIONS:
                    high_idx = self.HIGH_ACTIONS.index(high_act)
                    # 将最近的动作历史编码到特征中，越近的权重越高
                    action_history_features[high_idx * 3 + i] = 1.0
        
        # 添加动作计数信息
        action_count_features = np.zeros(len(self.HIGH_ACTIONS))
        for action, count in self.action_count.items():
            if action in self.HIGH_ACTIONS:
                idx = self.HIGH_ACTIONS.index(action)
                action_count_features[idx] = min(count / 5.0, 1.0)  # 归一化到[0,1]

        # 添加动作效果信息
        action_effect_features = np.zeros(len(self.HIGH_ACTIONS))
        for action, effect in self.recent_action_effects.items():
            if action in self.HIGH_ACTIONS:
                idx = self.HIGH_ACTIONS.index(action)
                # 归一化到[-1, 1]范围
                action_effect_features[idx] = np.clip(effect, -1.0, 1.0)


        # 添加归一化的步数，让模型知道时间在流逝
        normalized_step_count = float(self.step_count / self.max_steps)
        
        # 添加总执行动作数量（归一化）
        total_executed_actions = sum(self.action_count.values())
        
        # 添加最近性能变化的移动平均
        recent_perf_changes = self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history
        avg_recent_perf_change = np.mean(recent_perf_changes) if recent_perf_changes else 0.0

        return np.array([
            *quality_one_hot,
            cur_issue_severity,
            # perf_change,
            normalized_step_count,  # 替换了原来的float(self.step_count / self.max_steps)
            *action_oh,
            # normalized_cost,
            k_norm,
            # *action_history_features,
            # *action_count_features,
            # *action_effect_features,
            # normalized_total_actions,      # 添加总执行动作数量
            # avg_recent_perf_change        # 添加最近性能变化的移动平均
        ], dtype=np.float32)

    def _state_low(self, macro_action):
        if not self.numeric_cols:
            return np.zeros(8, dtype=np.float32)
        
        # 获取详细的违规率信息
        rates, missing_rates, outlier_rates, outlier_predict_labels, violation_rates, details_rates = self._get_issue_rates()
        
        # 解析详细的违规率信息
        avg_missing_rates, avg_outlier_rates, avg_violation_rates = rates
        avg_row_violation_rates, avg_speed_violation_rates, avg_accel_violation_rates = details_rates[0:3]
        avg_variance_violation_rates, avg_od_violation_rates, avg_dd_violation_rates = details_rates[3:6]
        
        target_oh = np.zeros(len(self.ISSUE_MAP))
        if macro_action in self.ISSUE_MAP:
            target_oh[self.ISSUE_MAP[macro_action]] = 1.0
        total_stats = []
        total_structure = []
        
        
        # 输入数据始终为numpy数组，处理三维数据 (n_samples, n_timesteps, n_features)
        for sample_idx in range(self.cur_train.shape[0]):
            sample_data = self.cur_train[sample_idx, :, :]

            stats = {'mean': [], 'std': [], 'skew': [], 'kurtosis': []}
            structure = []

            for col_idx in self.numeric_cols:
                col_data = sample_data[:, col_idx]
                # 移除nan值
                data = col_data[~np.isnan(col_data)]
                if len(data) >= 2:
                    stats['mean'].append(np.mean(data))
                    stats['std'].append(np.std(data))
                    stats['skew'].append(self._skew(data))
                    stats['kurtosis'].append(self._kurtosis(data))
                # 处理结构度量
                filled_data = np.nan_to_num(col_data)  # 简单处理缺失值
                structure.append(calculate_structure_metric(filled_data))

            sample_avg_stats = [np.mean(v) if v else 0.0 for v in stats.values()]
            sample_avg_smoothness = np.mean(structure) if structure else 0.0

            total_stats.append(sample_avg_stats)
            total_structure.append(sample_avg_smoothness)

        avg_stats = np.mean(total_stats, axis=0) if total_stats else np.zeros(4)
        avg_smoothness = np.mean(total_structure) if total_structure else 0.0

        # 添加详细的违规率信息到状态向量中
        return np.array([
            *target_oh, 
            *avg_stats, 
            avg_smoothness,
            avg_row_violation_rates,      # 行约束违规率
            avg_speed_violation_rates,    # 速度约束违规率
            avg_accel_violation_rates,    # 加速度约束违规率
            avg_variance_violation_rates, # 方差约束违规率
            avg_od_violation_rates,       # OD约束违规率
            avg_dd_violation_rates        # DD约束违规率
        ], dtype=np.float32)

    def _low_level_clean_flexible(self, macro, low_action):
        if not self.numeric_cols:
            return self.cur_train, 0.0
            
        # 获取检测结果 - 优先使用缓存的结果，如果没有则重新检测
        if hasattr(self, 'cached_detection_results') and self.cached_detection_results is not None:
            missing_rates, outlier_rates, outlier_predict_labels, violation_rates = self.cached_detection_results
        else:
            # 如果没有缓存结果，重新检测
            _, missing_rates, outlier_rates, outlier_predict_labels, violation_rates, _ = self._get_issue_rates()
            
        before_data = self.cur_train.copy()

        if macro == 'MISSING':
            self.cur_train = call_imputer(low_action, self.cur_train, missing_rates, self.missing_mask)  #missing_rates是Detector得到的
        elif macro == 'ANOMALY':
            self.cur_train = call_outlier_modifier(low_action, self.cur_train, outlier_predict_labels, outlier_rates, threshold=0)
        elif macro == 'CONSTRAINT':
            # 使用默认的约束处理程序，让智能体学习在状态信息基础上选择合适的low_action
            self.cur_train = call_constraints_handler(low_action, self.cur_train, violation_rates, self.constraints, concentrated=self.concentrated)

        # 清除缓存以确保下次调用_get_issue_rates时重新检测
        self.cached_detection_results = None

                
        # 计算变化量
        total_change = 0.0
        num_cells = 0
        if self.cur_train.dtype.kind in 'fc':
            total_change = np.sum(np.abs(self.cur_train - before_data))
            num_cells = self.cur_train.size
            
        average_change = total_change / num_cells if num_cells else 0.0
        return self.cur_train, average_change


    def step_low(self, macro_action, low_action):
        self.step_count += 1
        # 记录执行的low_action，以便在检测连续重复动作时使用
        self.last_low_action = low_action
        # 清除之前的缓存结果，因为我们即将修改数据
        self.cached_detection_results = None
        initial_rates, _, _, _, _, _ = self._get_issue_rates()
        initial_rate = initial_rates[self.ISSUE_MAP.get(macro_action, 0)]
        
        # 输入数据始终为numpy数组
        before_cols = self.cur_train[:, :, self.numeric_cols].copy()
        # 计算清洗前的结构度量
        before_structure = self._calculate_overall_structure(self.cur_train)

        self.cur_train, _ = self._low_level_clean_flexible(macro_action, low_action)

        final_rates, _, _, _, _, _ = self._get_issue_rates()
        final_rate = final_rates[self.ISSUE_MAP.get(macro_action, 0)]
        
        # 计算清洗后的结构度量
        after_structure = self._calculate_overall_structure(self.cur_train)

        # 归一化三个奖励分量，避免量纲影响
        structure_scale = max(abs(before_structure), abs(after_structure), 1e-6)
        R_Structure = (after_structure - before_structure) / structure_scale

        diff = self.cur_train[:, :, self.numeric_cols] - before_cols
        num_cells = max(diff.size, 1)
        R_Conservative = np.sqrt(np.sum(diff ** 2)) / np.sqrt(num_cells)

        R_Local_Signal = np.clip(initial_rate - final_rate, -1.0, 1.0)

        low_reward_components = {
            'structure_component': MU_1 * R_Structure,
            'conservative_component': -MU_2 * R_Conservative,
            'local_signal_component': MU_3 * R_Local_Signal,
        }
        low_reward_components = {
            key: (0.0 if np.isnan(value) else value)
            for key, value in low_reward_components.items()
        }
        R_L = sum(low_reward_components.values())
        
        # Check if maximum steps reached
        done = self.step_count > self.max_steps
        return self._state_low(macro_action), R_L, done

    def step_high(self, macro_action, low_reward, is_final=False):
        self.last_macro_action = macro_action
        self.prev_issue_severity = self.current_issue_severity
        step_start_time = time.time()

        if is_final or self.step_count > self.max_steps:
            terminal_rates, _, _, _, _, _ = self._get_issue_rates()
            terminal_issue_severity = float(np.sum(terminal_rates))
            terminal_quality_reward = np.clip(1.0 - terminal_issue_severity, -1.0, 1.0)
            terminal_quality_improvement = terminal_quality_reward - self.initial_terminal_quality_reward
            R_H = LAMBDA_GRAD * np.clip(terminal_quality_improvement, -1.0, 1.0)

            info = {
                'terminal_quality_reward': terminal_quality_reward,
                'terminal_quality_improvement': terminal_quality_improvement,
                'initial_terminal_quality_reward': self.initial_terminal_quality_reward,
                'issue_severity': terminal_issue_severity,
                'best_k': self.best_k,
            }
            return self._state_high(), R_H, True, info

        # 获取当前的指标率
        current_rates, _, _, _, _, _ = self._get_issue_rates()
        current_missing_rate = current_rates[0]
        current_outlier_rate = current_rates[1]
        current_violation_rate = current_rates[2]
        current_issue_severity = float(np.sum(current_rates))
        if self.prev_issue_severity is None:
            self.prev_issue_severity = current_issue_severity
        self.current_issue_severity = current_issue_severity
        
        # 计算指标的改进 - 使用上一次的值减去当前值
        # 由于这些指标越小越好，改进值为上一次值 - 当前值
        missing_improvement = getattr(self, 'previous_missing_rate', current_missing_rate) - current_missing_rate
        outlier_improvement = getattr(self, 'previous_outlier_rate', current_outlier_rate) - current_outlier_rate
        violation_improvement = getattr(self, 'previous_violation_rate', current_violation_rate) - current_violation_rate
        
        # 更新上一次的指标值
        self.previous_missing_rate = current_missing_rate
        self.previous_outlier_rate = current_outlier_rate
        self.previous_violation_rate = current_violation_rate

        self._update_best_k()


        # 只有在聚类任务中才计算k_stability，forecast和classification任务中设置为0
        if self.task_type == 'clustering':
            k_stability = 1.0 if len(self.k_history) >= K_STABILITY_WINDOW and len(set(self.k_history[-K_STABILITY_WINDOW:])) == 1 else 0.0
        else:
            k_stability = 0.0

        issue_improvement = missing_improvement + outlier_improvement + violation_improvement

        time_inc = time.time() - step_start_time
        R_Cost = max(0.0, time_inc)
              
        # 添加指标改进的奖励
        R_Issue = issue_improvement

        # 归一化高层奖励分量，避免量纲影响
        R_Issue_N = np.clip(R_Issue, -1.0, 1.0)
        R_Cost_N = min(R_Cost / max(MAX_COST_NORM, 1e-6), 1.0)
        low_reward_N = np.clip(low_reward, -1.0, 1.0)
        

        
        # 更新动作计数和记录
        if macro_action in self.action_count:
            self.action_count[macro_action] += 1
        else:
            self.action_count[macro_action] = 1
            
        # 更新动作效果记录
        self.recent_action_effects[macro_action] = issue_improvement
        
        # 记录当前动作和性能变化（在计算惩罚之前记录）
        # 为了正确检测连续重复的(macro_action, low_action)组合，我们现在保存完整的动作对
        if hasattr(self, 'last_low_action'):
            full_action = (macro_action, self.last_low_action)
        else:
            full_action = (macro_action, low_reward)  # fallback if last_low_action is not set
        self.action_history.append(full_action)
        self.performance_history.append(issue_improvement)
        
        # 保持历史记录长度合理
        if len(self.action_history) > 10:
            self.action_history.pop(0)
            self.performance_history.pop(0)
        
        # 检查是否需要对重复动作施加惩罚 - 在当前步骤就计算惩罚
        extra_penalty = 0.0
        # 检查是否存在连续3次重复的动作且性能增益微小
        if len(self.action_history) >= 3:
            # 获取最近的3次动作和性能变化
            last_three_actions = self.action_history[-3:]
            last_three_performances = self.performance_history[-3:]
            
            # 检查是否最后3次的完整动作组合完全相同，且性能提升都很小
            if (len(last_three_actions) == 3 and 
                all(action == last_three_actions[0] for action in last_three_actions) and 
                all(abs(perform) <= 0.0005 for perform in last_three_performances)):
                # 这意味着当前动作（第3个）是连续第3次重复相同动作且性能提升都很小，施加惩罚
                extra_penalty = -5.0  # 对执行连续重复无效操作施加惩罚
                
        reward_components = {
            'cost_component': -LAMBDA_2 * R_Cost_N,
            'issue_component': LAMBDA_3 * R_Issue_N,
            'low_reward_component': LAMBDA_4 * low_reward_N,
            'k_stability_component': ALPHA * k_stability,
            'extra_penalty_component': extra_penalty,
        }
        reward_components = {
            key: (0.0 if np.isnan(value) else value)
            for key, value in reward_components.items()
        }
        R_H = sum(reward_components.values())

        self.total_cost += max(0, time_inc)

        info = {
            'issue_reward': reward_components['issue_component'],
            'cost_reward': reward_components['cost_component'],      # 时间成本奖励
            'low_rl_reward': reward_components['low_reward_component'],  # 来自低层的奖励
            'extra_penalty': reward_components['extra_penalty_component'],  # 额外惩罚
            'best_k': self.best_k,
            'issue_improvement': issue_improvement,
            'issue_severity': current_issue_severity,
            'missing_improvement': missing_improvement,
            'outlier_improvement': outlier_improvement,
            'violation_improvement': violation_improvement,
            'k_stability_reward': reward_components['k_stability_component']  # k稳定性奖励
        }
        return self._state_high(rates=current_rates), R_H, False, info

    def _update_best_k(self):
        # 仅在聚类任务中更新最佳聚类数
        if self.task_type != 'clustering':
            # 非聚类任务直接返回，不执行任何操作
            return
            
        # 输入数据始终为numpy数组
        X = self.cur_train[:, :, self.numeric_cols].copy()
        X = _to_aeon_ts_input(X)
            
        k, _ = auto_select_k(X, (K_MIN, K_MAX))
        self.k_history.append(k)
        if len(self.k_history) > K_STABILITY_WINDOW:
            self.k_history.pop(0)
        self.best_k = k
        # 同步模型 k
        if self.task_type == 'clustering':
            # 正确更新聚类模型的参数
            if hasattr(self.proxy_model, 'estimator') and hasattr(self.proxy_model.estimator, 'set_params'):
                self.proxy_model.estimator.set_params(n_clusters=k)
            if hasattr(self.final_model, 'estimator') and hasattr(self.final_model.estimator, 'set_params'):
                self.final_model.estimator.set_params(n_clusters=k)
            
            # 同步更新_model_params中的参数
            if hasattr(self.proxy_model, '_model_params') and 'estimator' in self.proxy_model._model_params:
                self.proxy_model._model_params['estimator'].set_params(n_clusters=k)
            if hasattr(self.final_model, '_model_params') and 'estimator' in self.final_model._model_params:
                self.final_model._model_params['estimator'].set_params(n_clusters=k)
        return k, _

    def reset(self):
        self.cur_train = self.dirty_data.copy()
        self.cur_label = self.label.copy() if self.label is not None else None
        self.step_count = 0
        self.last_macro_action = 'NONE'
        self.total_cost = 0.0
        self.k_history = []
        self.best_k = 2
        # 初始化缓存检测结果
        self.cached_detection_results = None

        # Reset action tracking
        self.action_history = []  # 记录(high_action, low_action)对
        self.performance_history = []  # 记录对应的性能变化
        self.action_count = {}  # 记录每个动作的执行次数
        self.recent_action_effects = {}  # 记录每个动作的近期效果

        # self.proxy_model = self._create_proxy_model()
        # self.final_model = self._create_final_model()

        # 移除不必要的预训练代码，因为train_and_evaluate会在每次评估时重新训练模型
        if self.task_type == 'clustering':
            self._update_best_k()

        # self.initial_proxy_perf, _ = train_and_evaluate(self.task_type, 'proxy', self.proxy_model, self)
        # self.initial_final_perf, _ = train_and_evaluate(self.task_type, 'final', self.final_model, self)
        self.last_test_perf = self.initial_proxy_perf
        self.prev_test_perf = self.initial_proxy_perf
        self.current_issue_severity = None
        self.prev_issue_severity = None

        print(f"初始代理模型性能: {self.initial_proxy_perf:.4f} | 初始最终模型性能: {self.initial_final_perf:.4f}")
        if self.task_type == 'clustering':
            print(f"初始最佳 k: {self.best_k}")
        return self._state_high()


# ==============================
# HRL 代理（不变）
# ==============================
class HRLAgent:
    def __init__(self, actions, state_dim, lr=0.12, gamma=0.93, eps_start=0.99, eps_end=0.01, eps_decay=150):
        self.actions = actions
        self.q = {}
        self.lr, self.gamma = lr, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.steps = 0
        # 使用独立的随机数生成器，避免受全局种子影响
        self.rng = np.random.default_rng()  # 使用None作为种子，每次生成不同的随机状态

    def _key(self, s):
        return tuple(np.round(s, 3))

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps / self.eps_decay)

class MacroAgent(HRLAgent):
    def update(self, trajectory):
        for s, a, r, s_next, done in reversed(trajectory):
            next_max_q = 0.0
            k_next = self._key(s_next)
            if k_next in self.q and not done:
                next_max_q = np.max(self.q[k_next])
            target = r + self.gamma * next_max_q
            k = self._key(s)
            if k not in self.q:
                self.q[k] = np.full(len(self.actions), 50.0)
            try:
                idx = self.actions.index(a)
                self.q[k][idx] += self.lr * (target - self.q[k][idx])
            except ValueError:
                logging.warning(
                    f"Macro action '{a}' not in action space {self.actions}. "
                    f"State: {s}. Skipping Q-update."
                )

class LowAgent(HRLAgent):
    def __init__(self, low_actions_map, state_dim, **kwargs):
        super().__init__([], state_dim, **kwargs)
        self.q = {macro: {} for macro in low_actions_map if macro != 'FINISH'}
        self.low_actions_map = low_actions_map
        # 也需要为LowAgent创建独立的随机数生成器
        self.rng = np.random.default_rng()

    def choose(self, state, macro_action):
        self.steps += 1
        low_actions = self.low_actions_map.get(macro_action, ['NONE'])
        key = self._key(state)
        if macro_action not in self.q:
            return self.rng.choice(low_actions)
        if key not in self.q[macro_action]:
            self.q[macro_action][key] = np.full(len(low_actions), 50.0)
        if self.rng.random() < self.epsilon():
            return self.rng.choice(low_actions)
        return low_actions[np.argmax(self.q[macro_action][key])]

    def update(self, trajectory, macro_action):
        if not trajectory or macro_action not in self.q:
            return
        s, a, r, s_next, done = trajectory[0]
        low_actions = self.low_actions_map.get(macro_action, ['NONE'])
        k = self._key(s)
        k_next = self._key(s_next)
        if k not in self.q[macro_action]:
            self.q[macro_action][k] = np.full(len(low_actions), 50.0)
        next_max_q = 0.0
        if k_next in self.q[macro_action] and not done:
            next_max_q = np.max(self.q[macro_action][k_next])
        target = r + self.gamma * next_max_q
        try:
            idx = low_actions.index(a)
            self.q[macro_action][k][idx] += self.lr * (target - self.q[macro_action][k][idx])
        except ValueError:
            logging.warning(
            f"Action '{a}' is not valid for macro-action '{macro_action}'. "
            f"Valid actions: {low_actions}. Skipping Q-update."
        )

def macro_agent_choose_with_rates(macro_agent, state, current_issue_rates, env=None):
    macro_agent.steps += 1
    key = macro_agent._key(state)
    if key not in macro_agent.q:
        macro_agent.q[key] = np.zeros(len(macro_agent.actions))

    missing_rate = current_issue_rates[ISSUE_MAP['MISSING']]
    print(f"当前缺失率: {missing_rate:.4f}")
    if missing_rate > 0.001:
        return 'MISSING'

    # 增加初期完全随机探索的可能性
    if macro_agent.steps < 8:  # 前8步更倾向于随机探索
        explorable = [a for a in macro_agent.actions if a != 'FINISH']
        macro_action = macro_agent.rng.choice(explorable)
        return macro_action
    
    
    # 基于计数的探索_bonus
    # 原有的代码从这里开始...
    exploration_bonus = np.zeros(len(macro_agent.actions))
    if env and hasattr(env, 'action_count'):
        total_actions = sum(env.action_count.values()) + 1  # 避免除零
        for i, action in enumerate(macro_agent.actions):
            count = env.action_count.get(action, 0)
            # 使用计数的倒数作为探索_bonus，鼓励探索较少选择的动作
            exploration_bonus[i] = 1.0 / np.sqrt(count + 1)
    
    # 检查是否需要强制更换工具 - 当同一个(high_action, low_action)组合连续执行且没有明显指标改进时
    if env:
        has_repeated, repeated_action = _has_consecutive_repeated_actions_with_minimal_gain(env)
        if has_repeated:
            # 强制选择其他动作
            current_action = repeated_action[0]  # 当前的high action
            explorable = [a for a in macro_agent.actions if a != 'FINISH' and a != current_action]
            if explorable:
                print(f"强制更换工具: 连续执行 {repeated_action} 两次且无指标改进，强制选择其他工具")
                return macro_agent.rng.choice(explorable)

    if macro_agent.rng.random() < macro_agent.epsilon():
        explorable = [a for a in macro_agent.actions if a != 'FINISH' and a != 'MISSING']
        return macro_agent.rng.choice(explorable)

    q_values = macro_agent.q[key].copy()
    
    # 如果有动作计数信息，对执行次数较多的动作施加惩罚
    if env and hasattr(env, 'action_count'):
        penalty_factor = 0.1  # 惩罚因子
        for i, action in enumerate(macro_agent.actions):
            if action in env.action_count and env.action_count[action] > 2:
                q_values[i] -= penalty_factor * env.action_count[action]
    
    # 添加探索_bonus
    exploration_weight = 0.1  # 探索权重
    q_values = q_values + exploration_weight * exploration_bonus
    
    # 屏蔽已知无效的动作
    if env and hasattr(env, 'recent_action_effects'):
        for i, action in enumerate(macro_agent.actions):
            if action in env.recent_action_effects and env.recent_action_effects[action] < -0.1:
                # 如果最近效果很差，降低其Q值
                q_values[i] -= 2.0
    
    if 'FINISH' in macro_agent.actions and macro_agent.steps < 5:
        q_values[macro_agent.actions.index('FINISH')] = -np.inf
    return macro_agent.actions[np.argmax(q_values)]

def low_agent_choose_override(low_agent, state, macro_action, env=None):
    """
    重写low agent的选择逻辑，增加重复动作检测
    """
    low_agent.steps += 1
    low_actions = low_agent.low_actions_map.get(macro_action, ['NONE'])
    key = low_agent._key(state)
    if macro_action not in low_agent.q:
        return low_agent.rng.choice(low_actions)
    if key not in low_agent.q[macro_action]:
        low_agent.q[macro_action][key] = np.zeros(len(low_actions))
    
    # 检查是否需要强制更换工具 - 当连续三次执行相同的low action且没有指标改进时
    if env and len(env.action_history) >= 3:
        # 检查最近三次动作是否都是相同的(macro_action, low_action)组合且指标改进很小
        last_three_actions = env.action_history[-3:]
        last_three_performances = env.performance_history[-3:]
        
        # 检查是否最近三次执行的都是相同的macro_action和low_action，且指标改进都很小
        same_actions = all(
            action[0] == macro_action and action[1] in low_actions 
            for action in last_three_actions
        )
        small_improvements = all(
            abs(perform) <= 0.0005 
            for perform in last_three_performances
        )
        
        if same_actions and small_improvements:
            # 强制选择其他low action
            current_low_action = last_three_actions[-1][1]  # 获取当前的low action
            explorable = [a for a in low_actions if a != current_low_action]
            if explorable:
                print(f"强制更换低级工具: 连续执行 {macro_action}-{current_low_action} 三次且无指标改进，强制选择其他工具")
                return low_agent.rng.choice(explorable)
    
    # 基于计数的探索_bonus
    exploration_bonus = np.zeros(len(low_actions))
    if env and hasattr(env, 'action_history'):
        # 统计当前low action的执行次数
        action_counts = {}
        for _, low_act in env.action_history:
            if low_act in low_actions:
                action_counts[low_act] = action_counts.get(low_act, 0) + 1
        
        # 计算探索_bonus
        total_actions = sum(action_counts.values()) + 1
        for i, action in enumerate(low_actions):
            count = action_counts.get(action, 0)
            exploration_bonus[i] = 1.0 / np.sqrt(count + 1)
    
    if low_agent.rng.random() < low_agent.epsilon():
        return low_agent.rng.choice(low_actions)
    
    q_values = low_agent.q[macro_action][key].copy()
    
    # 添加探索_bonus
    exploration_weight = 0.1
    q_values = q_values + exploration_weight * exploration_bonus
    
    # 如果某些动作效果不好，降低其Q值
    if env and hasattr(env, 'action_history') and hasattr(env, 'performance_history') and len(env.action_history) >= 1:
        # 查看最近几次动作的效果
        recent_history = list(zip(env.action_history[-5:], env.performance_history[-5:]))
        action_effects = {}
        for (high_act, low_act), perf in recent_history:
            if high_act == macro_action and low_act in low_actions:
                if low_act not in action_effects:
                    action_effects[low_act] = []
                action_effects[low_act].append(perf)
        
        # 对效果差的动作进行惩罚
        for i, action in enumerate(low_actions):
            if action in action_effects:
                avg_effect = np.mean(action_effects[action])
                if abs(avg_effect) <= 0.0005:  # 几乎没有指标改进的动作
                    q_values[i] -= 1.0  # 适度降低Q值
                if avg_effect < -0.1:
                    q_values[i] -= 2.0  # 明显负面效果的动作
    
    return low_actions[np.argmax(low_agent.q[macro_action][key])]

def train_hrl_agents(env, macro_agent, low_agent, n_episodes):
    high_reward_history = []
    for episode in range(n_episodes):
        state_h = env.reset()
        done = False
        high_trajectory = []
        episode_high_reward = 0

        # # 重置agents的steps计数器
        # macro_agent.steps = 0
        # low_agent.steps = 0

        while not done:
            rates, _, _, _, _, _ = env._get_issue_rates()
            macro_action = macro_agent_choose_with_rates(macro_agent, state_h, rates, env)
            if env.step_count >= env.max_steps:
                macro_action = 'FINISH'

            if macro_action == 'FINISH':
                env.step_count += 1
                next_state_h, reward_h, done, info = env.step_high(macro_action, 0.0, is_final=True)
                print("\n")
                print(f"Episode {episode+1}/{n_episodes} - FINAL TASK COMPLETED")
                print(f"  Terminal Quality Reward: {info.get('terminal_quality_reward', 0.0):+.4f}")
                print(f"  Terminal Quality Improvement: {info.get('terminal_quality_improvement', 0.0):+.4f}")
                print(f"  Issue Severity: {info.get('issue_severity', 0.0):.4f}")
                print("\n")
                high_trajectory.append((state_h, macro_action, reward_h, next_state_h, done))
                episode_high_reward += reward_h
                state_h = next_state_h
                break


            state_l = env._state_low(macro_action)
            # 使用重写的选择函数
            low_action = low_agent_choose_override(low_agent, state_l, macro_action, env)
            next_state_l, reward_l, low_done = env.step_low(macro_action, low_action)
            low_trajectory = [(state_l, low_action, reward_l, next_state_l, False)]
            low_agent.update(low_trajectory, macro_action)

            start_state_h = state_h
            next_state_h, reward_h, high_done, info = env.step_high(macro_action, reward_l)
            
            done = low_done or high_done  # Combine done flags
            high_trajectory.append((start_state_h, macro_action, reward_h, next_state_h, done))
            episode_high_reward += reward_h
            state_h = next_state_h

            issue_improvement = info.get('issue_improvement', 0.0)
            extra_penalty = info.get('extra_penalty', 0.0)
            print("\n")
            print(f"Episode {episode+1}/{n_episodes} - Step: {env.step_count:2d} "
                  f"- High Action: {macro_action:<10} "
                  f"- Low Action: {low_action:<20} "
                  f"- Reward: {reward_h:+6.3f} "
                f"- Issue: {info.get('issue_severity', 0):.4f} "
                f"- ΔIssue: {issue_improvement:+.4f} "
                  f"- Penalty: {extra_penalty:+.4f} "
                  f"- best_k: {info.get('best_k', env.best_k)}")
            print(f"  [详细Reward分解] Issue:{info.get('issue_reward', 0):+.3f}, "
                  f"LowRL:{info.get('low_rl_reward', 0):+.3f}, "
                  f"Cost:{info.get('cost_reward', 0):+.3f}")
            print(f"  [改进指标] Missing:{info.get('missing_improvement', 0):+.4f}, "
                  f"Outlier:{info.get('outlier_improvement', 0):+.4f}, "
                  f"Violation:{info.get('violation_improvement', 0):+.4f}")
            print("\n")

        macro_agent.update(high_trajectory)
        high_reward_history.append(episode_high_reward)

    return high_reward_history


# ==============================
# 主函数
# ==============================
def main(dirty_data, label=None, task_type='forecast', detector=None, constraints=None, concentrated=True, max_steps=20, n_episodes=10):
    '''
    label: 训练集标签（classification 任务需要）
    task_type: 'classification', 'forecast', 'clustering'
    '''

    if detector is None:
        raise ValueError("Detector must be provided.")
    if constraints is None:
        raise ValueError("Constraints must be provided.")

    start_time = time.time()

    env = RLCleanEnvironment(dirty_data=dirty_data, label=label, task_type=task_type, 
                            detector=detector, constraints=constraints, concentrated=concentrated, max_steps=max_steps, train_ratio=0.7)   
    
    # 然后获取状态维度
    state_h_dim = env._state_high().shape[0]
    state_l_dim = env._state_low('MISSING').shape[0]

    macro_agent = MacroAgent(actions=env.HIGH_ACTIONS, state_dim=state_h_dim, lr=0.1, gamma=0.9)
    low_agent = LowAgent(low_actions_map=env.LOW_ACTIONS_MAP, state_dim=state_l_dim, lr=0.2, gamma=0.85)

    print("\n--- 开始 HRL 训练")
    _ = train_hrl_agents(env, macro_agent, low_agent, n_episodes=n_episodes)
    print("--- 训练完成 ---")

    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")

    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")

    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    # 计算总时间成本
    total_time_cost = time.time() - start_time
    print(f"总时间成本: {total_time_cost:.2f} 秒")
    
    # 返回清洗后的数据、时间成本以及训练好的策略模型
    return env.cur_train, total_time_cost, (macro_agent, low_agent)



if __name__ == '__main__':

    # 获取原始数据
    
    # type = 'forecast'
    # dataset_name = 'ETTh1'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=0.2)
    
    # type = 'forecast'
    # dataset_name = 'IDF_OilTemp'
    # data = load_original_data(dataset_name, 'clean', rate=1)
    # label_2_train = None



    type = 'classification'
    dataset_name = 'Libras'
    data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
    data, label_2_train = sample_data_by_rate(data, label_2_train, rate=0.5)

    # type = 'classification'
    # dataset_name = 'Handwriting'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
    # data, label_2_train = sample_data_by_rate(data, label_2_train, rate=0.2)


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
        0.3,
        ["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
        covered_attrs=range(data.shape[-1]),
    )

    dirty_data, error_mask = dm.get_dirty_data_restored()



    # 挖掘约束，检测异常
    miner_type = True # True: 使用统一的约束条件, False: 每个sample使用不同的约束条件
    detector = Detector(dm, miner_type=miner_type)
    missing_rate, outlier_rate, outlier_predict_labels, violation_rates, constraints, data_2_repair= detector.detect_all()
 
    print("data_2_repair", data_2_repair.shape)



    
    task_type = 'classification'
    model_save_path = f"/home/yyy/TSC/TSClean/AutoClean/Error_Cleaner/saved_models/wo_tasks/{task_type}_{dataset_name}_wo_tasks_5.pkl"

    # 训练清洗模型
    repaired_data, time_cost, trained_agents = main(data_2_repair, label_2_train, task_type, detector, constraints, 
                                                    concentrated=miner_type, max_steps=20, n_episodes=5)
    save_trained_agents(trained_agents, model_save_path)


    # # 调用模型清洗
    # repaired_data, time_cost = apply_cleaning_from_saved_model(
    # model_path=model_save_path,
    # dirty_data=data_2_repair,
    # label=label_2_train,
    # task_type=task_type,
    # detector=detector,
    # constraints=constraints,
    # concentrated=miner_type,
    # max_steps=20
    # )



    # 评估修复效果
    print("repaired_data:",repaired_data.shape)
    label_2_evaluate = error_mask.astype(int) # (n_samples,n_timestamps,n_features)
    label_2_evaluate[:, :, 0] = 0
    
    # 使用评估函数评估清洗效果
    # eval_results = evaluate_cleaning_effectiveness(data_2_repair, clean_data, label_2_evaluate, repaired_data, time_cost)
    eval_results = evaluate_cleaning_effectiveness(repaired_data, clean_data, data_2_repair, time_cost)
    print("\n清洗效果评估结果:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

    print("dataset name:", dataset_name)
    print("task type:", task_type)



    # # 提取第一列
    # col1_a = clean_data[0][:, 0]
    # col1_b = repaired_data[0][:, 0]

    # # 找出不相等的位置（布尔掩码）
    # diff_mask = col1_a != col1_b

    # # 获取行索引（哪里不同）
    # diff_indices = np.where(diff_mask)[0]

    # print("不同的行索引:", diff_indices)
    # print("对应值 (clean_data vs repaired_data):")
    # for i in diff_indices:
    #     print(f"  行 {i}: {col1_a[i]} vs {col1_b[i]}")





    
