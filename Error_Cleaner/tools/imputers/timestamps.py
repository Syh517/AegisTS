import numpy as np
import pandas as pd

def impute_timestamps(data_missing: np.ndarray):
    """
    修复时间戳列的缺失值。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :return: 修复后的时间戳列 numpy 数组 (N x T)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    n_samples, n_timesteps, n_features = data_missing.shape
    imputed_timestamps = np.zeros((n_samples, n_timesteps))
    
    # 对每个样本进行处理
    for i in range(n_samples):
        timestamps = data_missing[i, :, 0]  # 取第一个特征列（时间戳列）

        if np.isnan(timestamps).any():
            # 判断是否可以转换为 datetime 类型
            try:
                ts = pd.to_datetime(timestamps, errors="raise")
                ts_series = pd.Series(ts)
                imputed_ts = ts_series.interpolate(method="time").values
            except Exception:
                ts_series = pd.Series(timestamps)
                imputed_ts = ts_series.interpolate(method="linear").values
        else:
            imputed_ts = timestamps

        imputed_timestamps[i] = imputed_ts

    return imputed_timestamps