import pandas as pd
import numpy as np
import os

Train_Datasets_List = ['CalIt2', 'cicids', 'creditcard', 'Daphnet', 'Daphnet', 'GECCO', 'GECCO', 'GHL', 'GutenTAG', 'metro', 'MITDB', 'MSL', 'OPPORTUNITY', 'PSM', 'room-occupancy', 'SMAP', 'SMD', 'SVDB', 'swan']



def pick_continuous_subset_with_anomalies(df, subset_size=10000, random_state=42):
    """
    从原数据集中挑选一个连续片段子集，长度约 subset_size，保证包含异常点。
    """
    np.random.seed(random_state)
    n_rows = len(df)
    if n_rows <= subset_size:
        return df.copy()  # 数据本身就小
    
    # 所有可能窗口起始索引
    start_indices = range(0, n_rows - subset_size + 1)
    
    # 记录每个窗口的异常数量
    window_anomaly_counts = []
    labels = df['is_anomaly'].values
    for start in start_indices:
        end = start + subset_size
        count = np.sum(labels[start:end] == 1)
        if count > 0:
            window_anomaly_counts.append((start, count))
    
    if not window_anomaly_counts:
        raise ValueError("整个数据集中没有足够异常点，无法找到包含异常的连续窗口。")
    
    # 挑选异常数最多的窗口，也可以随机选一个窗口
    window_anomaly_counts.sort(key=lambda x: x[1], reverse=True)
    best_start = window_anomaly_counts[0][0]
    best_end = best_start + subset_size
    
    subset_df = df.iloc[best_start:best_end].copy().reset_index(drop=True)
    print(f"选中窗口 [{best_start}:{best_end}]，包含异常 {window_anomaly_counts[0][1]} 个")
    return subset_df


if __name__ == "__main__":

    data_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets/'
    reset_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets_Subsets/'
    os.makedirs(reset_direc, exist_ok=True) 

    subset_size = 10000

    for dataset in Train_Datasets_List:
        dataset_dir = os.path.join(data_direc, dataset)
        reset_dir = os.path.join(reset_direc, dataset)
        os.makedirs(reset_dir, exist_ok=True)
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
            savepath = os.path.join(reset_dir, file)
            filename = file.replace('_val.csv', '')
            print("filename:", filename)

            # try:
            df = pd.read_csv(filepath).dropna()

            if df.size == 0 or len(df) == 0:
                print(f"[Skip] Empty data: {filepath}")
                continue
            
            if len(df) > subset_size:
                subset = pick_continuous_subset_with_anomalies(df, subset_size=subset_size)
            else:
                subset = df

            subset.to_csv(savepath, index=False)

