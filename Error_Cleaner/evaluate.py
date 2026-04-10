import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, silhouette_score, 
    calinski_harabasz_score, mean_squared_error, precision_score, recall_score
)


def normalize(data):
    """
    Normalize the data using min-max normalization
    """
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (data - min_vals) / range_vals


def evaluate_cleaning_effectiveness(dirty_data, clean_data, label_2_evaluate, repaired_data, time_cost=0.0):
    """
    评估数据清洗效果
    参数:
    dirty_data: 原始脏数据
    clean_data: 真实的干净数据
    label_2_evaluate: 错误标签 (1表示错误，0表示正确)
    repaired_data: 修复后的数据
    time_cost: 清洗过程的时间成本（秒）
    
    返回:
    dict: 包含各种评估指标的字典
    """
    # 确保输入数据是三维的
    if len(dirty_data.shape) != 3:
        raise ValueError("dirty_data must be 3-dimensional (n_samples, n_timestamps, n_features)")
    if len(repaired_data.shape) != 3:
        raise ValueError("repaired_data must be 3-dimensional (n_samples, n_timestamps, n_features)")
    if len(clean_data.shape) != 3:
        raise ValueError("clean_data must be 3-dimensional (n_samples, n_timestamps, n_features)")
    
    # 初始化结果字典
    results = {}
    
    # 计算RMSE (Root Mean Square Error)
    # 使用真实干净数据计算RMSE
    rmse = np.sqrt(mean_squared_error(clean_data.flatten(), repaired_data.flatten()))
    results['RMSE'] = rmse
    
    # 计算MNAD (Mean Normalized Absolute Distance)
    # Normalize the data along the timestamp axis
    normalized_repaired = normalize(repaired_data)
    normalized_clean = normalize(clean_data)
    
    # Calculate the mean normalized absolute distance
    mnad = np.mean(np.abs(normalized_repaired - normalized_clean))
    results['MNAD'] = mnad
    
    # 计算RRA (Relative Repair Accuracy)
    # 这里我们假设错误标签标记了所有被注入的错误
    # 首先我们需要确定哪些点被修复了（在有错误的地方进行了更改）
    diff_original_repaired = np.abs(dirty_data - repaired_data)
    repaired_positions = diff_original_repaired > 1e-6  # 非常小的阈值，表示有变化
    
    # RRA = (正确修复的数量) / (总错误数量)
    # 正确修复: 在错误位置进行了修复 (repaired_positions & label_2_evaluate)
    # 总错误数量: label_2_evaluate中1的总数
    # 对于三维数据 (n_samples, n_timestamps, n_features)，我们在所有维度上求和
    correctly_repaired = np.sum(repaired_positions & (label_2_evaluate == 1))
    total_errors = np.sum(label_2_evaluate == 1)
    
    rra = correctly_repaired / total_errors if total_errors > 0 else 0.0
    results['RRA'] = rra
    
    # 计算Precision, Recall, F1
    # 将多维数组压平以便计算
    repaired_flat = repaired_positions.flatten()
    label_flat = label_2_evaluate.flatten()
    
    if len(np.unique(label_flat)) > 1:  # 确保有正负两类样本
        precision = precision_score(label_flat, repaired_flat, zero_division=0)
        recall = recall_score(label_flat, repaired_flat, zero_division=0)
        f1 = f1_score(label_flat, repaired_flat, zero_division=0)
    else:
        precision = recall = f1 = 0.0
    
    results['Precision'] = precision
    results['Recall'] = recall
    results['F1'] = f1
    
    # 记录Time Cost
    results['TimeCost'] = time_cost
    
    return results