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
import random, argparse
from sklearn.preprocessing import MinMaxScaler
from Error_Detection.detectors.evaluation.metrics import get_metrics
from Error_Detection.detectors.utils.slidingWindows import find_length_rank
from Error_Detection.detectors.model_wrapper import *
from Error_Detection.detectors.HP_list import algo_HP_dict

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


Unsupervise_AD_Pool = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'Sub_PCA', 'PCA', 'HBOS', 'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'FFT', 'NORMA',
                       'MovingAverage', 'EWMA', 'SavGol', 'Kalman', 'LOESS', 'HoltWinters', 'RollingMedian',
                       'ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM',
                       'SARAD', 'SensitiveHUE', 'SensorSCAN', 'CATCH', 'DADA',]

test = ['CalIt2', 'Daphnet']

test_AD = ['Sub_IForest', 'IForest']



def run_detectors0(datasets, models, data_direc, result_save_dir, verbose=True):
    """
    遍历每个数据集，再遍历每个模型，运行异常检测并收集性能指标。

    Args:
        datasets (list): 数据集名称列表（如 'SMD', 'MSL'）
        models (list): 模型名称列表
        data_direc (str): 数据根目录，结构应为: {data_direc}/{dataset}/{file}_test.csv
        result_save_path (str): 结果保存路径（CSV）
        verbose (bool): 是否打印中间信息

    Returns:
        results_df (pd.DataFrame): 包含 dataset, filename, model, metric1, metric2...
    """
    results_dict = {}  # {dataset_filename: {model: VUS-PR}}
    config_headers = []

    for dataset in datasets:
        dataset_dir = os.path.join(data_direc, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Dataset directory not found: {dataset_dir}")
            continue

        # 查找所有 _val.csv 文件
        selected_files = [f for f in os.listdir(dataset_dir) if f.endswith('_val.csv')]
        if len(selected_files) == 0:
            print(f"[Warning] No test files found in {dataset_dir}")
            continue

        for file in selected_files:
            filepath = os.path.join(dataset_dir, file)
            filename = file.replace('_val.csv', '')
            config_headers.append(filename)

            if filename not in results_dict:
                print("filename:", filename)
                results_dict[filename] = {}

            # try:
            df = pd.read_csv(filepath).dropna()
            data = df.iloc[:, 1:-1].values.astype(float)
            label = df.iloc[:, -1].values.astype(int)

            if data.size == 0 or len(data) == 0:
                print(f"[Skip] Empty data: {filepath}")
                continue


            slidingWindow = find_length_rank(data, rank=1)
            print(f"Processing {dataset}/{filename} | Data: {data.shape}, Label: {label.shape}")

            for model_name in models:
                if model_name not in algo_HP_dict:
                    print(f"[Skip] No HP defined for model: {model_name}")
                    continue

                hp = algo_HP_dict[model_name]


                
                output = run_Unsupervise_AD(model_name, data, **hp)

                # 归一化异常分数到 [0,1]
                if output.ndim == 1:
                    output = output.reshape(-1, 1)
                output = MinMaxScaler((0, 1)).fit_transform(output).ravel()

                # 获取评估指标
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+3*np.std(output)))
                results_dict[filename][model_name] = evaluation_result['VUS-PR']

                if verbose:
                    print(f"{model_name} | F1: {evaluation_result['F1']:.4f}, "
                        f"VUS-PR: {evaluation_result['VUS-PR']:.4f}")
                    # print('Evaluation Result: ', evaluation_result)

            # except Exception as e:
            #     print(f"[Error] Failed to load {filepath}: {str(e)}")
    
    # 转成 DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df.index.name = 'Dataset'
    results_df.reset_index(inplace=True)

    
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    result_save_path = path.join(result_save_dir, 'performance_matrix_vus_pr.csv')

    # 保存 CSV
    results_df.to_csv(result_save_path, index=False)
    print(f"[Info] Results saved to {result_save_path}")


def run_detectors(datasets, models, data_direc, result_save_dir, verbose=True):
    """
    遍历数据集和模型，运行异常检测，结果按宽表格式直接写入CSV。
    """
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    result_save_path = os.path.join(result_save_dir, 'performance_matrix.csv')

    # 每次运行都覆盖旧文件，先写表头
    header = ["Dataset"] + models
    with open(result_save_path, "w") as f:
        f.write(",".join(header) + "\n")


    for dataset in datasets:
        dataset_dir = os.path.join(data_direc, dataset)
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Dataset directory not found: {dataset_dir}")
            continue

        selected_files = [f for f in os.listdir(dataset_dir) if f.endswith('_val.csv')]
        if len(selected_files) == 0:
            print(f"[Warning] No test files found in {dataset_dir}")
            continue

        for file in selected_files:
            filepath = os.path.join(dataset_dir, file)
            filename = file.replace('_val.csv', '')

            try:
                df = pd.read_csv(filepath).dropna()
                data = df.iloc[:, 1:-1].values.astype(float)
                label = df.iloc[:, -1].values.astype(int)
                
                print("NaN count in raw data:", df.isna().sum().sum())

                if data.size == 0:
                    print(f"[Skip] Empty data: {filepath}")
                    continue



                slidingWindow = find_length_rank(data, rank=1)
                print(f"Processing {dataset}/{filename} | Data: {data.shape}, Label: {label.shape}")

                row_results = [filename]  # 第一列 Dataset 名称

                for model_name in models:
                    if model_name not in algo_HP_dict:
                        print(f"[Skip] No HP defined for model: {model_name}")
                        row_results.append("")  # 没有结果时留空
                        continue

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
                    row_results.append(f"{score:.6f}")

                    if verbose:
                        print(f"{model_name} | F1: {evaluation_result['F1']:.4f}, "
                        f"VUS-PR: {evaluation_result['VUS-PR']:.4f}")

                # 写入 CSV
                with open(result_save_path, "a") as f:
                    f.write(",".join(row_results) + "\n")

            except Exception as e:
                print(f"[Error] Failed to process {filepath}: {str(e)}")

    print(f"[Info] Results continuously saved to {result_save_path}")


if __name__ == '__main__':
    data_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/'
    result_save_dir = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results_val/'
    run_detectors(Train_Datasets_List, Unsupervise_AD_Pool, data_direc, result_save_dir, verbose=True)
    # run_detectors(test, test_AD, data_direc, result_save_dir, verbose=True)


    # ## ArgumentParser
    # parser = argparse.ArgumentParser(description='Running Detectors')
    # parser.add_argument('--filename', type=str, default='SMD_machine-1-2_test.csv')
    # parser.add_argument('--data_direc', type=str, default='/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/')
    # parser.add_argument('--save', type=bool, default=False)
    # parser.add_argument('--AD_Name', type=str, default='IForest')
    # args = parser.parse_args()

    # df = pd.read_csv(args.data_direc + args.filename).dropna()
    # data = df.iloc[:, 1:-1].values.astype(float)
    # # print(data)
    # label = df.iloc[:, -1].values.astype(int)

    # slidingWindow = find_length_rank(data, rank=1)
    # # print('slidingWindow: ', slidingWindow)



    # Optimal_Det_HP = algo_HP_dict[args.AD_Name]

    # print("data shape: ", data.shape)
    # print("label shape: ", label.shape)


    # if args.AD_Name in Unsupervise_AD_Pool:
    #     output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
    #     print("output:",output)
    #     print("output shape: ", output.shape)
    #     print("label shape: ", label.shape)
    # else:
    #     raise Exception(f"{args.AD_Name} is not defined")
    

    # if isinstance(output, np.ndarray):
    #     output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
    #     evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+3*np.std(output)))
    #     print('Evaluation Result: ', evaluation_result)
    # else:
    #     print(f'At {args.filename}: '+output)

