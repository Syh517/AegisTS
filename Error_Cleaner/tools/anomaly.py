import numpy as np
import pandas as pd
import importlib
from typing import Any, Union
import warnings

from Error_Detection.ts_column import clean_timestamp_column_with_regression
warnings.filterwarnings("ignore")

# 基于平滑的方法
from Error_Cleaner.tools.outlier_modifiers.moving_average import moving_average_repair
from Error_Cleaner.tools.outlier_modifiers.autoregressive import autoregressive_repair
from Error_Cleaner.tools.outlier_modifiers.arma import arma_repair
from Error_Cleaner.tools.outlier_modifiers.kalman_filter import kalman_filter_repair
from Error_Cleaner.tools.outlier_modifiers.interpolation import interpolation_repair
from Error_Cleaner.tools.outlier_modifiers.state_space import state_space_repair
from Error_Cleaner.tools.outlier_modifiers.trajectory_simplification import trajectory_simplification_repair
from Error_Cleaner.tools.outlier_modifiers.exponential_smoothing import exponential_smoothing_repair
from Error_Cleaner.tools.outlier_modifiers.svr import svr_repair

# 基于统计的方法
from Error_Cleaner.tools.outlier_modifiers.maximum_likelihood import maximum_likelihood_repair
from Error_Cleaner.tools.outlier_modifiers.bayesian_model import bayesian_model_repair
from Error_Cleaner.tools.outlier_modifiers.markov_model import markov_model_repair
from Error_Cleaner.tools.outlier_modifiers.hmm import hmm_repair
from Error_Cleaner.tools.outlier_modifiers.smurf import smurf_repair
from Error_Cleaner.tools.outlier_modifiers.spatio_temporal_prob_model import spatio_temporal_prob_model_repair
from Error_Cleaner.tools.outlier_modifiers.em import em_repair
from Error_Cleaner.tools.outlier_modifiers.relationship_network import relationship_network_repair
from Error_Cleaner.tools.outlier_modifiers.gaussian_mixture import gaussian_mixture_repair
from Error_Cleaner.tools.outlier_modifiers.imr import imr_repair
from Error_Cleaner.tools.outlier_modifiers.akane import akane_repair
# from Error_Cleaner.tools.outlier_modifiers.ols import ols_repair
from Error_Cleaner.tools.outlier_modifiers.ols_residual import ols_residual_repair

# 基于异常检测/深度模型的方法
from Error_Cleaner.tools.outlier_modifiers.dbscan import dbscan_repair
from Error_Cleaner.tools.outlier_modifiers.lof_neighbor_mean import lof_neighbor_mean_repair
from Error_Cleaner.tools.outlier_modifiers.abnormal_sequence_interpolation import abnormal_sequence_interpolation_repair
from Error_Cleaner.tools.outlier_modifiers.window_regression import window_regression_repair
from Error_Cleaner.tools.outlier_modifiers.clustering_kmeans import clustering_kmeans_repair
from Error_Cleaner.tools.outlier_modifiers.wavelet_denoise import wavelet_denoise_repair
from Error_Cleaner.tools.outlier_modifiers.lstm_sequence import lstm_sequence_repair
from Error_Cleaner.tools.outlier_modifiers.cnn import cnn_repair
from Error_Cleaner.tools.outlier_modifiers.gan import gan_repair
from Error_Cleaner.tools.outlier_modifiers.tranad import tranad_repair
from Error_Cleaner.tools.outlier_modifiers.imdiffusion import imdiffusion_repair
from Error_Cleaner.tools.outlier_modifiers.cae_m import cae_m_repair

def call_outlier_modifier(method: str, dirty_data: np.ndarray, label: np.ndarray, anomaly_rates: np.ndarray, threshold: float = 0.3, **kwargs) -> Any:
    """
    label: shape (num_samples, n_timestamps), binary values indicating anomalies
    anomaly_rates: shape (num_samples,), float values indicating anomaly rates per sample
    threshold: minimum anomaly rate for a sample to be repaired
    """    

    # 保存原始时间戳列（第一列）
    original_timestamps = dirty_data[:, :, 0].copy()
    
    # 构造完整的模块路径和函数名
    module_path = f"Error_Cleaner.tools.outlier_modifiers.{method}"
    function_name = f"{method}_repair"
    
    try:
        # 动态导入模块
        module = importlib.import_module(module_path)
        # 获取函数
        func_obj = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"无法导入函数 '{method}': {str(e)}")
    
    # 创建修复数据的副本
    repaired_data = dirty_data.copy()
    
    # 确定哪些样本需要修复（异常率高于阈值）
    samples_to_repair = anomaly_rates >= threshold

    # 输出调试信息
    print(f"正在使用清洗工具: {method}")

    
    # 只对异常率高的样本进行修复
    if np.any(samples_to_repair):
        # 提取需要修复的样本
        dirty_data_subset = dirty_data[samples_to_repair]
        label_subset = label[samples_to_repair]
        
        # 对这些样本进行修复
        repaired_subset = func_obj(dirty_data_subset, label_subset, **kwargs)
        
        # 恢复时间戳列（第一列）为原始值
        repaired_subset[:, :, 0] = original_timestamps
        
        # 将修复后的数据放回原数组中对应的位置
        repaired_data[samples_to_repair] = repaired_subset
    
    # 恢复时间戳列（第一列）为原始值
    repaired_data[:, :, 0] = original_timestamps
    
    return repaired_data




anomaly_methods = [
    # 基于平滑的方法
    "moving_average",
    "autoregressive",
    "arma",
    "kalman_filter",
    "interpolation",
    "state_space",
    "trajectory_simplification",
    "exponential_smoothing",
    "svr",
    
    # 基于统计的方法
    "maximum_likelihood",
    "bayesian_model",
    "markov_model",
    "hmm",
    "smurf",
    "spatio_temporal_prob_model",
    "em",
    "relationship_network",
    "gaussian_mixture",
    "imr",
    "akane",
    "ols_residual",
    
    # 基于异常检测/深度模型的方法
    "dbscan",
    "lof_neighbor_mean",
    "abnormal_sequence_interpolation",
    "window_regression",
    "clustering_kmeans",
    "wavelet_denoise",
    "lstm_sequence",
    "cnn",
    "gan",
    "tranad",
    "imdiffusion",
    "cae_m",
]



if __name__ == "__main__":

    from Error_Injection.injector import DataManager
    from Datasets.load_dataset import load_single_dataset

    type = 'forecast'
    dataset_name = 'ETTh1'
    data, label =load_single_dataset(type, dataset_name, rate=0.1)

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
        covered_attrs=range(data.shape[-1]),
    )

    dirty_data, error_mask = dm.get_dirty_data_restored()
    data_2_repair = np.copy(dirty_data)
    data_2_repair = clean_timestamp_column_with_regression(data_2_repair) 
    label = error_mask.any(axis=2).astype(int)
    anomaly_rates = np.mean(label, axis=1)
    print("data_2_repair", data_2_repair.shape)
    print("label:", label.shape)
    print("anomaly_rates:", anomaly_rates.shape)



    # 只修复异常率大于0.1的样本
    data_repaired = call_outlier_modifier("ols_residual", data_2_repair, label, anomaly_rates, threshold=0.1)
    print("data_repaired:", data_repaired.shape)



    clean_data = dm.clean_data_raw.copy()
    # 提取第一列
    col1_a = clean_data[0][:, 0]
    col1_b = data_repaired[0][:, 0]

    # 找出不相等的位置（布尔掩码）
    diff_mask = col1_a != col1_b

    # 获取行索引（哪里不同）
    diff_indices = np.where(diff_mask)[0]

    print("不同的行索引:", diff_indices)
    print("对应值 (data_2_repair vs repaired_data):")
    for i in diff_indices:
        print(f"  行 {i}: {col1_a[i]} vs {col1_b[i]}")



    # methods = [ols_residual_repair,]


    # results = {}
    # for func in methods:
    #     print(f"\n 运行 {func.__name__} ...")
    #     try:
    #         data_repaired = func(data_abnormal, label)
    #         if not isinstance(data_repaired, pd.DataFrame):
    #             raise ValueError("返回类型错误，必须是 DataFrame")
    #         # 简单的效果验证：修复点的数值波动应减小
    #         diff = np.mean(np.abs(data_repaired - data_abnormal))
    #         results[func.__name__] = diff
    #         print(f"{func.__name__} 完成，平均改变量: {diff:.4f}")
    #     except Exception as e:
    #         print(f"{func.__name__} 失败: {e}")
    #         results[func.__name__] = None

    # print("\n=== 测试结果汇总 ===")
    # for name, score in results.items():
    #     print(f"{name:35s} => {'Fail' if score is None else f'{score:.4f}'}")