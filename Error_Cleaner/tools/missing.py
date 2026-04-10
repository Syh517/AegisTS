import numpy as np
import pandas as pd
import importlib
from typing import Any, Union
import warnings

from Error_Detection.ts_column import clean_timestamp_column_with_regression
warnings.filterwarnings("ignore")

from Error_Cleaner.tools.imputers.interpolation import impute_interpolation
from Error_Cleaner.tools.imputers.moving_average import impute_moving_average
from Error_Cleaner.tools.imputers.ar import impute_ar
from Error_Cleaner.tools.imputers.kalman_filter import impute_kalman_filter
from Error_Cleaner.tools.imputers.gmm_em import impute_gmm_em
from Error_Cleaner.tools.imputers.hmm import impute_hmm
from Error_Cleaner.tools.imputers.svr_univariate import impute_svr_univariate



missing_methods = [
    "interpolation",
    "moving_average",
    "ar",
    "kalman_filter",
    "gmm_em",
    "hmm",
    "svr_univariate",
]

def break_consecutive_values(data: np.ndarray) -> np.ndarray:
    """
    Add slight variations to consecutive identical values to prevent long sequences
    of identical values in columns after imputation.
    
    Args:
        data: 3D array of shape (N, T, D) where N is samples, T is timesteps, D is features
        
    Returns:
        Modified data with consecutive identical values broken
    """
    data_modified = data.copy()
    n_samples, n_timesteps, n_features = data_modified.shape
    
    # Process each sample and feature independently
    for i in range(n_samples):
        for j in range(n_features):
            column = data_modified[i, :, j]
            
            # Skip columns that are all NaN
            if np.all(np.isnan(column)):
                continue
                
            # Find consecutive identical values
            for t in range(1, n_timesteps):
                # Check if current and previous values are equal and not NaN
                if (not np.isnan(column[t]) and not np.isnan(column[t-1]) and 
                    abs(column[t] - column[t-1]) < 1e-10):  # Essentially equal
                    
                    # Add a small random variation
                    variation = np.random.normal(0, abs(column[t]) * 0.001 + 1e-8)
                    data_modified[i, t, j] = column[t] + variation
                    
    return data_modified


def call_imputer(method: str, dirty_data: np.ndarray, missing_rates: np.ndarray, missing_mask: np.ndarray = None, **kwargs) -> Any:
    """
    missing_rates: shape (num_samples,), float values indicating missing rates per sample
    missing_mask: shape (num_samples, time_steps, features), boolean mask indicating missing positions
    """

    # 保存原始时间戳列（第一列）
    original_timestamps = dirty_data[:, :, 0].copy()    
    
    # 支持的方法列
    if method not in missing_methods:
        raise ValueError(f"不支持的插补方法: {method}. 支持的方法: {missing_methods}")
    
    # 构造完整的模块路径和函数名
    module_path = f"Error_Cleaner.tools.imputers.{method}"
    function_name = f"impute_{method}"
    
    try:
        # 动态导入模块
        module = importlib.import_module(module_path)
        # 获取函数
        func_obj = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"无法导入函数 '{method}': {str(e)}")
    
    # 创建插补数据的副本，并将对应missing_mask标记为缺失的位置设置为NaN
    imputed_data = dirty_data.copy()
    
    if missing_mask is not None:
        # 将对应missing_mask标记为True的位置设置为NaN
        imputed_data = imputed_data.astype(float)  # 确保数据类型为float以支持NaN
        imputed_data[missing_mask] = np.nan
    
    # # 如果提供了missing_mask，则对所有标记为缺失的位置进行插补
    # # 否则基于missing_rates进行插补
    # if missing_mask is not None:
    #     # 检查哪些样本包含缺失值
    #     samples_to_impute = np.any(missing_mask, axis=(1, 2))
    # else:
    #     # 确定哪些样本需要插补（缺失率大于0）
    #     samples_to_impute = missing_rates > 0

    # # 输出调试信息
    # print(f"正在使用清洗工具: {method}")

    
    # # 只对包含缺失值的样本进行插补
    # if np.any(samples_to_impute):
    #     # 提取需要插补的样本
    #     dirty_data_subset = imputed_data[samples_to_impute]

        
    #     # 对这些样本进行插补
    #     imputed_subset = func_obj(dirty_data_subset, **kwargs)
        
    #     # 将插补后的数据放回原数组中对应的位置
    #     imputed_data[samples_to_impute] = imputed_subset

    # 考虑到数据中可能存在inf需要在不含缺失值的样本中，所以在插补方法内部一起处理了nan和inf
    imputed_data = func_obj(imputed_data, **kwargs)    
    
    # # Post-process to break consecutive identical values
    # imputed_data = break_consecutive_values(imputed_data)
        
    # 恢复时间戳列（第一列）为原始值
    imputed_data[:, :, 0] = original_timestamps

    return imputed_data


if __name__ == "__main__":

    from Error_Injection.injector import DataManager
    from Datasets.load_dataset import load_single_dataset
    from Error_Detection.Detector import Detector

    type = 'forecast'
    dataset_name = 'ETTh1'
    data, label = load_single_dataset(type, dataset_name)


    if type == 'forecast':
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        print(data.shape)
        dm = DataManager(data, abnormal_rate=0.1, task_type=type)
    else:
        print(data.shape, label.shape)
        dm = DataManager(data, label, abnormal_rate=0.1, task_type=type)

    dm.inject_errors(0.3, ["missing"], covered_attrs=range(data.shape[-1]))



    miner_type = True # True: 使用统一的约束条件, False: 每个sample使用不同的约束条件
    detector = Detector(dm)
    missing_rate1, outlier_rate, outlier_predict_labels, violation_rates, constraints= detector.detect_all()
    print("missing_rate_1",missing_rate1)


    dirty_data, error_mask = dm.get_dirty_data_restored()
    data_2_repair = np.copy(dirty_data)
    data_2_repair = clean_timestamp_column_with_regression(data_2_repair)
    # label = error_mask.any(axis=2).astype(int)
    # missing_rates = np.mean(label, axis=1)
    # print("data_2_repair", data_2_repair.shape)
    # print("missing_rates:", missing_rates)

    print("\n=== 调用插补方法 ===")
    imputed_data = call_imputer("interpolation", data_2_repair, missing_rate1)
    print("imputed_data:", imputed_data.shape)

    missing_rate2, outlier_rate, outlier_predict_labels, violation_rates, constraints= detector.detect_all(imputed_data)
    print("missing_rate_2",missing_rate2)


    clean_data = dm.clean_data_raw.copy()
    # 提取第一列
    col1_a = clean_data[0][:, 0]
    col1_b = imputed_data[0][:, 0]

    # 找出不相等的位置（布尔掩码）
    diff_mask = col1_a != col1_b

    # 获取行索引（哪里不同）
    diff_indices = np.where(diff_mask)[0]

    print("不同的行索引:", diff_indices)
    print("对应值 (clean_data vs repaired_data):")
    for i in diff_indices:
        print(f"  行 {i}: {col1_a[i]} vs {col1_b[i]}")