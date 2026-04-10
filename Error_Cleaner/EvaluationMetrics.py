import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def evaluate_cleaning_effectiveness(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray, time_cost: float):
    """
    Evaluate the effectiveness of cleaning by comparing the cleaned data with the original clean data.
    :param d_repair: The repaired data.
    :param d_clean: The origianal clean data.
    :param d_dirty: The dirty data.
    :return: The evaluation metrics.
    """
    if d_repair.ndim == 3:
        N, T, D = d_repair.shape
        d_repair = d_repair.reshape(-1, D)  # (N*T, D)
    if d_clean.ndim == 3:
        N, T, D = d_clean.shape
        d_clean = d_clean.reshape(-1, D)  # (N*T, D)
    if d_dirty.ndim == 3:
        N, T, D = d_dirty.shape
        d_dirty = d_dirty.reshape(-1, D)  # (N*T, D)

    # 检查整个数组是否有任何 NaN
    print(f"repair数据中是否存在 NaN: {np.isnan(d_repair).any()}")
    print(f"clean数据中是否存在 NaN: {np.isnan(d_clean).any()}")
    print(f"dirty数据中是否存在 NaN: {np.isnan(d_dirty).any()}")

    d_repair = np.nan_to_num(d_repair, nan=0.0)
    d_clean = np.nan_to_num(d_clean, nan=0.0)
    d_dirty = np.nan_to_num(d_dirty, nan=0.0)


    results = {}
    results['mse'] = mse(d_repair, d_clean)
    results['mnad'] = mnad(d_repair, d_clean)
    results['rra'] = rra(d_repair, d_clean, d_dirty)
    results['precision'], results['recall'] = precision_recall(d_repair, d_clean, d_dirty)
    results['mae_error'] = mae_error(d_repair, d_clean, d_dirty)
    results['time_cost'] = time_cost

    if results['precision'] and results['recall']:
        f1_score = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
        results['f1_score'] = f1_score
    else:
        results['f1_score'] = 0.0
        
    return results

def mse(d_repair: np.ndarray, d_clean: np.ndarray):
    # Assume that the input of the scaler is a DataFrame
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        # denominator = mean_squared_error(data_clean_scaled[:, i], data_dirty_scaled[:, i])
        # if denominator == 0:
            # continue
        sum += mean_squared_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim


def mnad(d_repair: np.ndarray, d_clean: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        # denominator = mean_absolute_error(data_clean_scaled[:, i], data_dirty_scaled[:, i])
        # if denominator == 0:
            # continue
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim


def rra(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    # Assume that the input of the scaler is a DataFrame
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_dirty_scaled = scaler.transform(d_dirty)
    data_repair_scaled = scaler.transform(d_repair)
    mae_rc = mnad(d_repair, d_clean)
    mae_rd = 0
    mae_cd = mnad(d_dirty, d_clean)
    dim = data_repair_scaled.shape[1]
    for i in range(dim):
        mae_rd += mean_absolute_error(data_dirty_scaled[:, i], data_repair_scaled[:, i])
    mae_rd /= dim
    return 1 - mae_rc / (mae_cd + mae_rd)


def precision_recall(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10000
    error_array = np.abs(d_dirty - d_clean) > bound
    repair_array = np.abs(d_dirty - d_repair) > bound
    benefit_repair_array = (np.abs(d_clean - d_repair) < np.abs(d_dirty - d_clean)) * repair_array
    tp = error_array * benefit_repair_array
    # print(np.sum(tp != 0), np.sum(repair_array != 0))
    return np.sum(tp != 0) / (np.sum(repair_array != 0)), np.sum(tp != 0) / (np.sum(error_array != 0))


def mae_error(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    d_repair = np.where(d_dirty == d_clean, d_clean, d_repair)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim