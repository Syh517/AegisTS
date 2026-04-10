import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

from Error_Cleaner.tools.constraint_violation_handlers.interface import clean_with_tool
from Error_Detection.Detector import Detector
from Error_Detection.ts_column import clean_timestamp_column_with_regression


constraint_methods=[
    'mtclean',
    'speedacc',
    'clean4mts',
    'mtcsc_n',
    'htd',
    'screen',
    'oneMilp',
    'speed_local',
    'speed_global',
    'speed_accel_local',
    'speed_accel_global',
    'variance',
]


def call_constraints_handler(method: str, dirty_data: np.ndarray, violation_rates: Dict[str, Any], constraints, concentrated=True, **kwargs) -> Any:
    """
    处理三维dirty_data，将其转换为二维DataFrame格式，并调用clean_with_tool进行清洗
    
    Args:
        method: 清洗方法名称
        dirty_data: 三维numpy数组，形状为(n_samples, n_timesteps, n_features)
        violation_rates: 违规率信息
        constraints: 约束条件字典
        **kwargs: 其他参数
    
    Returns:
        清洗后的数据
    """
    # 获取数据维度信息
    n_samples, n_timesteps, n_features = dirty_data.shape
    
    # 存储所有清洗后的数据
    repaired_samples = []

    # 保存原始时间戳列（第一列）
    original_timestamps = dirty_data[:, :, 0].copy()
    
    # 对每个样本进行处理
    for i in range(n_samples):
        # 取出单个样本并转换为二维格式
        sample_data = dirty_data[i]  # 形状为(n_timesteps, n_features)

        if concentrated:  #使用统一的约束条件
            sample_constraints = constraints
        else:
            sample_constraints = constraints[i]

        
        # 生成列名
        columns = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
        
        # 创建DataFrame
        # 第一列作为timestamp，其余列作为特征列
        df_data = pd.DataFrame(sample_data, columns=columns)
        
        # 调用clean_with_tool进行清洗
        repaired_sample = clean_with_tool(method, df_data, sample_constraints)
        
        # 将清洗后的数据转换回numpy数组格式并存储
        repaired_samples.append(repaired_sample.values)
    
    # 将所有清洗后的样本重新组合成三维数组
    repaired_data = np.stack(repaired_samples, axis=0)
    
    # 恢复时间戳列（第一列）为原始值
    repaired_data[:, :, 0] = original_timestamps
    
    return repaired_data


if __name__ == "__main__":

    from Error_Injection.injector import DataManager
    from Datasets.load_dataset import load_single_dataset

    type = 'forecast'
    dataset_name = 'ETTh1'
    data, label =load_single_dataset(type, dataset_name)

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


    detector = Detector(dm)
    missing_rate, outlier_rate, outlier_predict_labels, violation_rates, constraints= detector.detect_all()

    dirty_data, error_mask = dm.get_dirty_data_restored()
    data_2_repair = np.copy(dirty_data)
    data_2_repair = clean_timestamp_column_with_regression(data_2_repair) 
    print("data_2_repair", data_2_repair[:10])

    repaired_data = call_constraints_handler("clean4mts", dirty_data, violation_rates, constraints, concentrated=True)
    print("repaired_data:",repaired_data[:10])

    missing_rate, outlier_rate, outlier_predict_labels, violation_rates, constraints= detector.detect_all(repaired_data)