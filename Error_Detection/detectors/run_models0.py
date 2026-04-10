# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd
import numpy as np
import torch
from os import path
import random
import argparse
from sklearn.preprocessing import MinMaxScaler
from Error_Detection.detectors.evaluation.metrics import get_metrics
from Error_Detection.detectors.utils.slidingWindows import find_length_rank
from Error_Detection.detectors.model_wrapper import *
from Error_Detection.detectors.HP_list import algo_HP_dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def downsample_data(data, label, downsampling_factor=1):
    """
    对时间序列数据进行降采样以减少计算负担，同时保持时序特性
    
    Args:
        data: 输入数据 (n_timestamps, n_features)
        label: 标签数据 (n_timestamps,)
        downsampling_factor: 降采样因子，例如2表示每隔一个点取一个点
        
    Returns:
        降采样后的数据和标签
    """
    if downsampling_factor <= 1:
        return data, label
    
    # 按固定间隔采样，保持时间序列特性
    indices = np.arange(0, len(data), downsampling_factor)
    return data[indices], label[indices]


# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

Train_Datasets_List = ['CalIt2', 'cicids', 'creditcard', 'Daphnet', 'GECCO', 'GHL', 'GutenTAG', 'metro', 'MITDB', 'MSL', 'OPPORTUNITY', 'PSM', 'room-occupancy', 'SMAP', 'SMD', 'SVDB', 'swan']

# 定义小型数据集子集
Small_Datasets = ['CalIt2', 'Daphnet', 'GECCO', 'PSM', 'SMAP']

Unsupervise_AD_Pool = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'Sub_PCA', 'PCA', 'HBOS', 'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'FFT', 'NORMA',
                       'MovingAverage', 'EWMA', 'SavGol', 'Kalman', 'LOESS', 'HoltWinters', 'RollingMedian',
                       'ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM',
                       'SARAD', 'SensitiveHUE', 'SensorSCAN', 'CATCH', 'DADA',]

# 定义模型分类，便于选择
FAST_MODELS = ['ZScore', 'MAD', 'IQR', 'MovingAverage', 'EWMA']  # 快速模型
STAT_MODELS = ['ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM']  # 统计模型
ML_MODELS = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'Sub_PCA', 'PCA', 'HBOS', 'KNN', 'COPOD', 'CBLOF']  # 机器学习模型

test = ['CalIt2', 'Daphnet']

test_AD = ['Sub_IForest', 'IForest']




def process_model_on_dataset(args):
    """
    在单独的进程中处理单个模型和数据集的函数
    """
    model_name, dataset_name, file, data_direc, downsampling_factor = args
    try:
        filepath = os.path.join(data_direc, dataset_name, file)
        filename = file.replace('_val.csv', '')
        
        df = pd.read_csv(filepath).dropna()
        data = df.iloc[:, 1:-1].values.astype(float)
        label = df.iloc[:, -1].values.astype(int)
        
        # 数据降采样
        data, label = downsample_data(data, label, downsampling_factor)
        
        if data.size == 0:
            return filename, model_name, None

        slidingWindow = find_length_rank(data, rank=1)
        
        if model_name not in algo_HP_dict:
            return filename, model_name, None

        hp = algo_HP_dict[model_name]
        output = run_Unsupervise_AD(model_name, data, **hp)

        # 标准化异常分数
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        output = MinMaxScaler((0, 1)).fit_transform(output).ravel()

        # 计算指标
        evaluation_result = get_metrics(
            output,
            label,
            slidingWindow=slidingWindow,
            pred=output > (np.mean(output)+3*np.std(output))
        )

        score = evaluation_result["VUS-PR"]
        return filename, model_name, score
    except Exception as e:
        print(f"[Error] Failed to process {dataset_name}/{file} with {model_name}: {str(e)}")
        return file.replace('_val.csv', ''), model_name, None


def run_detectors(datasets, models, data_direc, result_save_dir, verbose=True, max_workers=None, downsampling_factor=1, max_files_per_dataset=None):
    """
    遍历数据集和模型，运行异常检测，结果按宽表格式直接写入CSV。
    添加并行处理支持以提高效率。
    """
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    result_save_path = os.path.join(result_save_dir, 'performance_matrix.csv')

    # 每次运行都覆盖旧文件，先写表头
    header = ["Dataset"] + models
    with open(result_save_path, "w") as f:
        f.write(",".join(header) + "\n")

    # 准备所有任务
    tasks = []
    for dataset in datasets:
        dataset_dir = os.path.join(data_direc, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Dataset directory not found: {dataset_dir}")
            continue

        # 查找所有 _val.csv 文件
        selected_files = [f for f in os.listdir(dataset_dir) if f.endswith('_val.csv')]
        if len(selected_files) == 0:
            print(f"[Warning] No validation files found in {dataset_dir}")
            continue
            
        # 如果指定了每个数据集的最大文件数，则限制文件数量
        if max_files_per_dataset is not None and len(selected_files) > max_files_per_dataset:
            selected_files = selected_files[:max_files_per_dataset]
            print(f"Limited to {max_files_per_dataset} files for dataset {dataset}")

        for file in selected_files:
            for model_name in models:
                tasks.append((model_name, dataset, file, data_direc, downsampling_factor))

    # 使用并行处理执行任务
    if max_workers is None:
        max_workers = min(32, mp.cpu_count())  # 默认使用最多32个进程，或CPU核心数
        
    print(f"Starting parallel processing with {max_workers} workers...")
    
    results_dict = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_model_on_dataset, task): task for task in tasks}
        
        # 收集结果
        for future in as_completed(future_to_task):
            filename, model_name, score = future.result()
            
            if filename not in results_dict:
                results_dict[filename] = {}
                
            results_dict[filename][model_name] = score if score is not None else ""
            
            if verbose and score is not None:
                print(f"{filename} - {model_name}: {score:.6f}")

    # 按数据集和文件名顺序写入结果
    for filename in sorted(results_dict.keys()):
        row_results = [filename]
        for model_name in models:
            score = results_dict[filename].get(model_name, "")
            if isinstance(score, float):
                row_results.append(f"{score:.6f}")
            else:
                row_results.append("")
        
        # 写入 CSV
        with open(result_save_path, "a") as f:
            f.write(",".join(row_results) + "\n")

    print(f"[Info] Results continuously saved to {result_save_path}")


if __name__ == '__main__':
    data_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/'
    result_save_dir = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results_val/'
    
    # 添加命令行参数支持
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--downsampling-factor', type=int, default=1, help='Data downsampling factor (>=1)')
    parser.add_argument('--max-files-per-dataset', type=int, default=None, help='Maximum number of files per dataset')
    parser.add_argument('--test', action='store_true', help='Run in test mode with small dataset')
    parser.add_argument('--small', action='store_true', help='Run on small subset of datasets')
    parser.add_argument('--model-set', choices=['all', 'fast', 'stat', 'ml'], default='all', 
                       help='Choose model set to run: all, fast, stat, ml')
    args = parser.parse_args()
    
    # 根据选择的模型集确定要运行的模型
    if args.model_set == 'fast':
        models_to_run = FAST_MODELS
    elif args.model_set == 'stat':
        models_to_run = STAT_MODELS
    elif args.model_set == 'ml':
        models_to_run = ML_MODELS
    else:
        models_to_run = Unsupervise_AD_Pool
    
    # 根据选择的数据集确定要运行的数据集
    if args.small:
        datasets_to_run = Small_Datasets
    elif args.test:
        datasets_to_run = test
    else:
        datasets_to_run = Train_Datasets_List
    
    run_detectors(datasets_to_run, models_to_run, data_direc, result_save_dir, verbose=True, 
                 max_workers=args.workers, downsampling_factor=args.downsampling_factor,
                 max_files_per_dataset=args.max_files_per_dataset)


