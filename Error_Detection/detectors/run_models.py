# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import random
from Error_Detection.detectors.evaluation.metrics import get_metrics
from Error_Detection.detectors.utils.slidingWindows import find_length_rank
from Error_Detection.detectors.model_wrapper import *
from Error_Detection.detectors.HP_list import algo_HP_dict
from collections import defaultdict
import csv
import tempfile
from concurrent.futures import as_completed, ProcessPoolExecutor  # 仅用于单进程时占位，后续删除

# ---------------- Seeding ----------------
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

# ---------------- Datasets & Models ----------------
Train_Datasets_List = ['CalIt2', 'cicids', 'creditcard', 'Daphnet', 'GECCO', 'GHL', 'GutenTAG', 'metro', 'MITDB',
                       'MSL', 'OPPORTUNITY', 'PSM', 'room-occupancy', 'SMAP', 'SMD', 'SVDB', 'swan']


Unsupervise_AD_Pool = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'Sub_PCA', 'PCA', 'HBOS', 'Sub_HBOS', 'KNN',
                       'Sub_KNN', 'KMeansAD', 'KMeansAD_U', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'FFT',
                       'NORMA', 'MovingAverage', 'EWMA', 'SavGol', 'Kalman', 'LOESS', 'HoltWinters', 'RollingMedian',
                       'ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM',
                       'SARAD', 'SensitiveHUE', 'SensorSCAN', 'CATCH', 'DADA']

# Unsupervise_AD_Pool = ['PCA', 'Sub_PCA', 'MAD', 'IForest', 'Mahalanobis', 'HotellingT2', 'Sub_IForest', 'ZScore', 'Sub_HBOS', 'LOESS',
#                         'KMeansAD', 'PCAStat', 'KMeansAD_U', 'CBLOF', 'EIF', 'KDE', 'COPOD', 'HBOS', 'Sub_LOF', 'EWMA', 'KNN','HoltWinters',
#                         'SavGol', 'Kalman', 'CATCH', 'MovingAverage', 'RobustPCA','SensitiveHUE', 'COF']

TSET_MODELS = ['SARAD', 'SensitiveHUE', 'SensorSCAN', 'CATCH', 'DADA']

MAX_SEQUENCE_LENGTH = 20000  # 仅超过此长度才截断

# ---------------- Model Processing ----------------
def process_model_on_dataset(args):
    model_name, dataset_name, file, data_direc = args
    filename = file.replace('_val.csv', '')

    try:
        filepath = os.path.join(data_direc, dataset_name, file)
        df = pd.read_csv(filepath).dropna()
        data_original = df.iloc[:, 1:-1].values.astype(float)
        label_original = df.iloc[:, -1].values.astype(int)

        # 仅大于 MAX_SEQUENCE_LENGTH 才截断
        if data_original.shape[0] > MAX_SEQUENCE_LENGTH:
            data = data_original[:MAX_SEQUENCE_LENGTH]
            label = label_original[:MAX_SEQUENCE_LENGTH]
        else:
            data = data_original
            label = label_original
            
        current_length = data.shape[0]
        while np.unique(label).shape[0] <= 1 and current_length < df.shape[0]:
            # 尝试增加5000个数据点
            current_length = min(current_length + 5000, df.shape[0])
            data = data_original[:current_length]
            label = label_original[:current_length]
                
        print(f"{filename} - {model_name}: {data.shape}, {label.shape},{np.unique(label)}")
        if np.unique(label).shape[0] <= 1:
            return filename, model_name, None

        if data.size == 0 or model_name not in algo_HP_dict:
            return filename, model_name, None

        slidingWindow = find_length_rank(data, rank=1)
        hp = algo_HP_dict[model_name].copy()

        output = run_Unsupervise_AD(model_name, data, **hp)

        # 归一化
        output = np.atleast_1d(output).astype(np.float64)
        out_min, out_max = output.min(), output.max()
        if out_max > out_min:
            output = (output - out_min) / (out_max - out_min)
        else:
            output = np.zeros_like(output)

        # 二值预测，防止全 0
        threshold = np.mean(output) + 3 * np.std(output)
        # threshold = 0.5
        pred = output > threshold


        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
        score = evaluation_result.get("VUS-PR", None)

        print(f"{filename} - {model_name}: {score}")
        return filename, model_name, score

    except Exception as e:
        print(f"[Error] {filename} with {model_name}: {str(e)}")
        return filename, model_name, None

# ---------------- Detector Runner ----------------
def run_detectors(datasets, models, data_direc, result_save_dir, max_files_per_dataset=None):
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    result_save_path = os.path.join(result_save_dir, 'performance_matrix.csv')
    
    # 始终确保文件存在且有正确的标题行
    with open(result_save_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset"] + models)

    print("Starting serial execution...")
    
    # 遍历所有数据集和文件
    for dataset in datasets:
        dataset_dir = os.path.join(data_direc, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Dataset directory not found: {dataset_dir}")
            continue

        selected_files = [f for f in os.listdir(dataset_dir) if f.endswith('_val.csv')]
        if max_files_per_dataset:
            selected_files = selected_files[:max_files_per_dataset]

        for file in selected_files:
            filename = file.replace('_val.csv', '')

            print(f"Processing dataset: {filename}")
            
            # 为当前数据集存储所有模型的结果
            current_dataset_results = {}
            
            # 在当前数据集上运行所有模型
            for model_name in models:
                try:
                    _, _, score = process_model_on_dataset((model_name, dataset, file, data_direc))
                    current_dataset_results[model_name] = score if score is not None else ""
                    print(f"{filename} - {model_name}: {score}")
                except Exception as e:
                    print(f"[Error] {filename} with {model_name}: {str(e)}")
                    current_dataset_results[model_name] = ""
            
            # 当前数据集的所有模型运行完毕，立即写入CSV
            with open(result_save_path, "a", newline='') as f:
                writer = csv.writer(f)
                row = [filename] + [current_dataset_results.get(m, "") for m in models]
                writer.writerow(row)
                print(f"Results for {filename} written to CSV.")

    print(f"✅ All results saved to: {result_save_path}")

# ---------------- Main ----------------
if __name__ == '__main__':
    # ---------------- 用户自定义变量 ----------------
    data_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/'
    result_save_dir = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results_test/'
    test_mode = True # 'all' or 'test'
    max_files_per_dataset = None


    # Train_Datasets_List = ['CalIt2', 'cicids', 'creditcard', 'Daphnet', 'GECCO', 'GHL', 'GutenTAG', 'metro', 'MITDB',
    #                    'MSL', 'OPPORTUNITY', 'PSM', 'room-occupancy', 'SMAP', 'SMD', 'SVDB', 'swan']
    if test_mode:
        models_to_run = ['SensitiveHUE']
        datasets_to_run = [ 'swan']
    else:
        models_to_run = Unsupervise_AD_Pool
        datasets_to_run = Train_Datasets_List

    run_detectors(
        datasets_to_run,
        models_to_run,
        data_direc,
        result_save_dir,
        max_files_per_dataset=max_files_per_dataset
    )
