import numpy as np
import pandas as pd
from Error_Injection.injector import DataManager
from Datasets.load_dataset import load_single_dataset, sample_data_by_rate
from sklearn.metrics import (
    roc_auc_score, f1_score, silhouette_score, 
    calinski_harabasz_score, mean_squared_error, precision_score, recall_score, accuracy_score
)

type = 'classification'
dataset_name = 'Handwriting'
data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
data, label_2_train = sample_data_by_rate(data, label_2_train, rate=0.2)


if type == 'forecast':
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    print(data.shape)
    dm = DataManager(data, abnormal_rate=0.1, task_type=type)
else:
    print(data.shape, label_2_train.shape)
    dm = DataManager(data, label_2_train, abnormal_rate=0.1, task_type=type)




# 注入错误
dm.inject_errors(
    0.3,
    ["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
    covered_attrs=range(data.shape[-1]),
)

dirty_data, error_mask = dm.get_dirty_data_restored()
data_2_repair = np.copy(dirty_data)
print("data_2_repair", data_2_repair.shape)

X = data_2_repair.transpose(0, 2, 1)  # 交换 axis=1 和 axis=2


# N, T, C = data_2_repair.shape

# # 1. 将每个样本的每个特征单独做 ffill + bfill
# cleaned = np.empty_like(data_2_repair)

# for i in range(N):  # 遍历每个样本
#     # 取出第 i 个样本: (T, C)
#     sample = data_2_repair[i]  # shape (T, C)
    
#     # 转为 DataFrame（列是特征，行是时间步）
#     df = pd.DataFrame(sample)  # shape (T, C)
    
#     # 前向填充 + 后向填充（处理开头的 NaN）
#     df_filled = df.fillna(method='ffill').fillna(method='bfill')
    
#     # 转回 numpy 并存入结果
#     cleaned[i] = df_filled.values

# # 最终得到清洗后的 (N, T, C) 数组
# X = cleaned.transpose(0, 2, 1)  # 如果你需要 (N, C, T) 格式给 aeon




########clustering test#########
from aeon.clustering.deep_learning import AEDCNNClusterer
from aeon.clustering.feature_based import Catch22Clusterer
from sklearn.cluster import KMeans


# # 20 个样本，4 个变量（通道），每个长度为 128
# data = np.random.randn(20, 4, 128)  # (n_samples, n_channels, n_timepoints)

# # 先定义底层聚类器，并指定 n_clusters
# estimator = KMeans(n_clusters=3, random_state=42)

# cluster = AEDCNNClusterer(
#     estimator=estimator,
#     n_epochs=50,
#     random_state=42,
#     batch_size=32
# )

# cluster =Catch22Clusterer(estimator=KMeans(n_clusters=2, random_state=42))

# X = np.nan_to_num(X)
# cluster.fit(X)  
# labels = cluster.predict(X)
# print(labels)



# if hasattr(cluster, '_network') and hasattr(cluster._network, 'encoder'):
#     print("Encoder device:", next(cluster._network.encoder.parameters()).device)

# X_test_2d = X.mean(axis=2)  # shape: (n_samples, n_channels)
# sil = silhouette_score(X_test_2d, labels)
# ch = calinski_harabasz_score(X_test_2d, labels)

# norm_sil = (sil + 1) / 2
# if ch <= 0:
#     norm_ch = 0.0
# else:
#     # 使用标准的逻辑函数形式，确保输出在(0,1)范围内
#     norm_ch = ch / (ch + 1.0)
# score = 0.6 * norm_sil + 0.4 * norm_ch
# score = float(np.clip(score, 0.0, 1.0))
# print(f"Silhouette Score: {sil:.4f}, Normalized: {norm_sil:.4f}")
# print(f"Calinski-Harabasz Score: {ch:.4f}, Normalized: {norm_ch:.4f}")
# print(f"Clustering Score: {score:.4f}")



########classification test#########
from aeon.datasets import load_arrow_head  # 示例数据集（单变量时间序列）
from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.convolution_based import MiniRocketClassifier
from sklearn.metrics import accuracy_score


# # 1. 加载数据（自动返回 3D 格式: (n_samples, n_channels, n_timepoints)）
# X_train, y_train = load_arrow_head(split="train")
# X_test, y_test = load_arrow_head(split="test")

# split = int(0.8 * X.shape[0])
# X_train = np.nan_to_num(X[:split])
# y_train = label_2_train[:split]
# X_test = np.nan_to_num(X[split:])
# y_test = label_2_train[split:]

# print("X_train shape:", X_train.shape)  # e.g., (36, 1, 251)
# print("y_train shape:", y_train.shape)  # e.g., (36,)

# # 2. 创建模型
# model = InceptionTimeClassifier(n_epochs=100, random_state=42)

# # model = MiniRocketClassifier(n_kernels=5000, random_state=42)

# # 3. 训练模型
# model.fit(X_train, y_train)

# # 4. 预测
# y_pred = model.predict(X_test)

# # 5. 评估
# acc = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {acc:.4f}")


########forecast test############
from aeon.regression.convolution_based import MiniRocketRegressor
from aeon.regression.deep_learning import InceptionTimeRegressor

from Error_Cleaner.DLiner import DLinear
from Error_Cleaner.LSTMForecast import LSTMForecast

# def to_supervised(data_3d, window_size, horizon):
#     """
#     将单条多变量时间序列 (1, T, F) 转换为监督学习格式 (N, window_size, F)
    
#     Parameters:
#     ----------
#     data_3d : np.ndarray
#         形状为 (1, n_timesteps, n_features) 的输入数据
#     window_size : int
#         历史窗口长度（用于预测）
#     horizon : int
#         预测步长（当前只支持 horizon=1，即预测下一步）

#     Returns:
#     -------
#     X : np.ndarray of shape (N, window_size, n_features)
#         输入特征
#     y : np.ndarray of shape (N, n_features)
#         目标值（下一个时间步的真实值）
#     """
#     if data_3d.shape[0] != 1:
#         raise ValueError("Forecast task expects batch size = 1")
    
#     series = data_3d.squeeze(0)  # (T, F)
#     T, F = series.shape
    
#     if T <= window_size:
#         # 数据太少，无法构造窗口
#         return np.empty((0, window_size, F)), np.empty((0, F))
    
#     X, y = [], []

#     for t in range(T - window_size - horizon):
#         X_window = series[t : t + window_size]                   # (window_size, F)
#         Y_future = series[t + window_size : t + window_size + horizon]  # (horizon, F)

#         X.append(X_window)
#         # y.append(Y_future.reshape(-1))  # flatten 成 (horizon*F,)
#         y.append(Y_future)
    
#     return np.array(X), np.array(y)  # (N, W, F), (N, H, F)



# # ===============================
# # 1. 构造模拟多变量时序数据
# # ===============================
# T = 17420
# D = 8
# data_3d = np.random.randn(1,T, D)  # 原始数据 (1,T, D)

# window_size = 500
# horizon = 100
# F = data_3d.shape[2]

# X, y = to_supervised(data_3d, window_size=window_size, horizon=horizon)
# print('X shape:', X.shape)
# print('y shape:', y.shape)
# # X: (N, 50, F)
# # y: (N, 10, F)

# # model = DLinear(seq_len=window_size, pred_len=horizon, num_features=F)
# model = LSTMForecast(input_dim=F, hidden_dim=64, pred_len=horizon)


# model.fit(X, y)
# y_pred = model.predict(X)  # (N, horizon, F)
# print('y_pred shape:', y_pred.shape)


import torch; print(torch.__version__)