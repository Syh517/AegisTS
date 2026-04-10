import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 假设缺失值用 np.nan 表示
def impute_interpolation(data_missing: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    使用插值法修复缺失值。
    对于多变量数据，逐列进行插值。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param method: 插值方法 ('linear', 'spline', 'nearest' 等)。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行插值
    for i in range(n_samples):
        # 转换为DataFrame以利用pandas的插值功能
        sample_df = pd.DataFrame(data_imputed[i])
        
        # pandas 的 interpolate 方法可以方便地处理 NaN 
        # 默认按列（变量）进行插值，保留了变量间的独立插值特性。
        sample_df = sample_df.interpolate(method=method, limit_direction='both', axis=0)
        
        # 如果数据开头或结尾仍有 NaN（取决于limit_direction），使用最近邻填充
        sample_df = sample_df.ffill().bfill()
        
        # 将处理后的数据存回
        data_imputed[i] = sample_df.values
    
    return data_imputed


def impute_moving_average(data_missing: np.ndarray, window: int = 5) -> np.ndarray:
    """
    使用滑动窗口平均法修复缺失值。
    注意：在有缺失值的情况下，简单的 MA 只能用于替换原有的缺失值，而不是预测。
    这里使用 rolling().mean()，结合 interpolate，作为一种平滑填充方式。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param window: 滑动窗口大小。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行处理
    for i in range(n_samples):
        # 转换为DataFrame以利用pandas的功能
        sample_df = pd.DataFrame(data_imputed[i])
        
        # 1. 计算滑动平均 (跳过 NaN)
        # rolling(window).mean() 忽略 NaN，直到窗口内有足够数量的非 NaN 值
        smoothed_data = sample_df.rolling(window=window, min_periods=1, center=True).mean()
        
        # 2. 用平滑后的值填充原始数据的缺失值
        sample_df = sample_df.fillna(smoothed_data)
        
        # 3. 如果开头或结尾仍有 NaN，退回到线性插值或最近邻填充（防止 NaN 链条过长）
        sample_df = sample_df.fillna(method='ffill').fillna(method='bfill')
        
        # 将处理后的数据存回
        data_imputed[i] = sample_df.values
    
    return data_imputed


from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import StandardScaler

def impute_ar(data_missing: np.ndarray, lags=1) -> np.ndarray:
    """
    对每个变量单独用 AR(lags) 插补缺失值
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param lags: AR模型的滞后阶数。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行处理
    for i in range(n_samples):
        sample_df = pd.DataFrame(data_imputed[i])
        
        for col in sample_df.columns:
            series = sample_df[col]
            nan_idx = series[series.isna()].index
            
            # 如果全是 NaN，跳过
            if series.dropna().empty:
                continue
            
            # 拟合 AR 模型（用已有值）
            series_filled = series.ffill().bfill() # 临时填充
            try:
                model = AutoReg(series_filled, lags=lags, old_names=False).fit()
                preds = model.predict(start=0, end=len(series)-1)
                sample_df.loc[nan_idx, col] = preds[nan_idx]
            except Exception as e:
                # 如果 AR 模型失败，用线性插值
                sample_df[col] = series.interpolate(limit_direction='both')
        
        # 将处理后的数据存回
        data_imputed[i] = sample_df.values
    
    return data_imputed


from pykalman import KalmanFilter

def impute_kalman_filter(data: np.ndarray, verbose=True) -> np.ndarray:
    """
    使用Kalman滤波器修复缺失值。
    :param data: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param verbose: 是否打印详细信息。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape

    # 对每个样本进行处理
    for i in range(n_samples):
        df = pd.DataFrame(data_imputed[i])

        if verbose:
            print(f"\n[Kalman 插补] 数据形状: {df.shape}")
            print("初始缺失率:")
            print(df.isna().mean().round(4))

        # ========== 阶段 1: 初步线性插值 ========== #
        df_interp = df.interpolate(method="linear", limit_direction="both")
        if df_interp.isna().any().any():
            df_interp = df_interp.fillna(method="ffill").fillna(method="bfill")
        if verbose:
            print(f"线性插值后仍存在 NaN: {df_interp.isna().any().any()}")

        # ========== 阶段 2: Kalman Filter 参数估计 ========== #
        df_values = df_interp.values
        n_dim = df_values.shape[1]

        try:
            kf = KalmanFilter(
                transition_matrices=np.eye(n_dim),
                observation_matrices=np.eye(n_dim),
                em_vars=[
                    "transition_covariance",
                    "observation_covariance",
                    "initial_state_mean",
                    "initial_state_covariance",
                ],
            )
            kf = kf.em(df_values, n_iter=20)

            # ========== 阶段 3: RTS 平滑 ========== #
            smoothed_state_means, _ = kf.smooth(df_values)
            df_smooth = pd.DataFrame(smoothed_state_means, columns=df.columns)

            # ========== 阶段 4: 替换缺失值 ========== #
            df_filled = df.copy()
            mask_missing = df.isna()
            df_filled[mask_missing] = df_smooth[mask_missing]

            # ========== 阶段 5: 二次补全（如首尾仍有 NaN） ========== #
            if df_filled.isna().any().any():
                if verbose:
                    print("Kalman 平滑结果仍存在缺失，执行二次线性补全 ...")
                df_filled = df_filled.interpolate(method="linear", limit_direction="both")

        except Exception as e:
            print("Kalman 滤波器训练失败，直接使用线性插值结果:", str(e))
            df_filled = df_interp

        # ========== 阶段 6: 输出统计信息 ========== #
        if verbose:
            nan_ratio = df_filled.isna().mean().round(6)
            print("\n插补完成。最终缺失率:")
            print(nan_ratio)
            if nan_ratio.sum() == 0:
                print("所有缺失值已成功补全。")
            else:
                print("仍存在缺失。")

        # 将处理后的数据存回
        data_imputed[i] = df_filled.values

    return data_imputed


from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

def impute_gmm_em(data_missing: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    使用高斯混合模型 (GMM) 和期望最大化 (EM) 算法修复缺失值。
    - GMM 用于对完整的观测数据进行聚类，EM 算法天然用于处理缺失数据。
    - 这里采用迭代方法：填充 -> 训练 GMM -> 重新填充。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param n_components: GMM组件数量。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行处理
    for i in range(n_samples):
        # 转换为DataFrame以利用pandas的功能
        sample_df = pd.DataFrame(data_imputed[i])
        data = sample_df.values
        T, D = data.shape
        sample_data_imputed = data.copy()
        
        # 1. 初始填充 (使用均值或线性插值)
        sample_df_filled = sample_df.fillna(method='ffill').fillna(method='bfill')
        sample_data_imputed = sample_df_filled.values
        
        nan_mask = np.isnan(data)
        
        # 2. 迭代 EM 过程
        for _ in range(5):  # 迭代次数
            # 训练 GMM 模型
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
            gmm.fit(sample_data_imputed)
            
            # 使用 GMM 重新估计缺失值 (E-step 的简化版)
            # 对于缺失的行，计算其属于每个高斯分量的后验概率 P(k|X_obs)，
            # 然后用这些分量的均值加权来估计缺失值。
            
            # 预测所有点的分量概率
            resp = gmm.predict_proba(sample_data_imputed)
            
            # 遍历缺失点并重新估计
            for t in range(T):
                if np.any(nan_mask[t]):
                    # 缺失点的估计值是所有高斯分量的均值的加权和
                    estimated_mean = np.dot(resp[t], gmm.means_)
                    sample_data_imputed[t, nan_mask[t]] = estimated_mean[nan_mask[t]]
                    
        # 将处理后的数据存回
        data_imputed[i] = sample_data_imputed
    
    return data_imputed


def impute_hmm(data_missing: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    使用隐马尔科夫模型 (HMM) 修复缺失值。
    - HMM 适用于序列数据，通过潜在状态来建模时序依赖。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param n_components: HMM状态数量。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行处理
    for i in range(n_samples):
        # 转换为DataFrame以利用pandas的功能
        sample_df = pd.DataFrame(data_imputed[i])
        data = sample_df.values
        T, D = data.shape
        nan_mask = np.isnan(data)
        
        # 1. 初始填充
        sample_df_filled = sample_df.fillna(method='ffill').fillna(method='bfill')
        data_imputed_sample = sample_df_filled.values

        try:
            # 2. 拟合 HMM 模型
            # GaussianHMM 假设观测值是多元高斯分布
            model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=5)
            model.fit(data_imputed_sample)

            # 3. 使用 Viterbi/前向-后向算法修复缺失值
            # 修复逻辑：对于每个缺失点，根据其所在时刻的最可能隐藏状态，
            # 用该隐藏状态对应的观测均值来填充缺失值。
            
            # 找到最可能的隐藏状态序列
            logprob, state_sequence = model.decode(data_imputed_sample, algorithm="viterbi")
            
            # 获取每个状态的观测均值
            means = model.means_
            
            # 4. 用 HMM 均值替换缺失值
            data_imputed_hmm = data_imputed_sample.copy()
            
            for t in range(T):
                if np.any(nan_mask[t]):
                    # 获取该时刻最可能的隐藏状态
                    most_likely_state = state_sequence[t]
                    # 获取该状态对应的观测均值
                    imputation_value = means[most_likely_state]
                    
                    # 仅替换缺失值
                    data_imputed_hmm[t, nan_mask[t]] = imputation_value[nan_mask[t]]

        except Exception as e:
            print(f"HMM 拟合失败 ({e})，退回到线性插值。")
            # 使用线性插值替代
            sample_df_interp = sample_df.interpolate(limit_direction='both')
            data_imputed_hmm = sample_df_interp.values

        # 将处理后的数据存回
        data_imputed[i] = data_imputed_hmm

    return data_imputed


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

def create_dataset(data, look_back=1):
    """
    创建 SVR 的时序数据集：输入为前 look_back 个时间步，输出为当前时间步。
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

def impute_svr_univariate(data_missing: np.ndarray, look_back: int = 5) -> np.ndarray:
    """
    使用支持向量回归 (SVR) 修复缺失值。
    - **注意：** SVR 是非时序模型，需要将时间序列转化为监督学习问题。
    - 这里采用**逐列（变量）**修复，并使用线性插值作为预填充。
    - 修复过程：针对每个变量，在缺失点前进行预测。
    :param data_missing: 待修复的 numpy 数组 (N x T x D)，N为样本数，T为时间步数，D为特征数。
    :param look_back: 用于预测的历史时间步数。
    :return: 修复后的 numpy 数组 (N x T x D)。
    """
    # 验证输入是3D数组
    assert len(data_missing.shape) == 3, "输入数据必须是三维numpy数组 (N x T x D)"
    
    data_imputed = data_missing.copy()
    n_samples, n_timesteps, n_features = data_imputed.shape
    
    # 对每个样本进行处理
    for i in range(n_samples):
        # 转换为DataFrame以利用pandas的功能
        sample_df = pd.DataFrame(data_imputed[i])
        
        # 1. 预填充（用于训练 SVR）
        sample_df_prefilled = sample_df.interpolate(method='linear', limit_direction='both')
        data_prefilled = sample_df_prefilled.values
        
        for d in range(sample_df.shape[1]): # 遍历每个变量
            series = sample_df.iloc[:, d].values
            series_prefilled = data_prefilled[:, d]
            
            # 2. 转化 SVR 数据集
            # X: (T-look_back) x look_back, Y: (T-look_back) x 1
            X_train, Y_train = create_dataset(series_prefilled, look_back)
            
            if len(X_train) == 0:
                continue
                
            # 将 X 展平为二维矩阵 (T-look_back) x look_back
            X_train = X_train.reshape(X_train.shape[0], -1) 
            
            # 3. 训练 SVR 模型
            # 使用 RBF 核，并加入标准化
            svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
            svr_model.fit(X_train, Y_train)
            
            # 4. 预测缺失值
            for t in range(look_back, len(series)):
                if np.isnan(series[t]):
                    # 构造输入特征 (t-look_back 到 t-1)
                    X_test = series_prefilled[t-look_back:t].reshape(1, -1)
                    
                    # 预测缺失值
                    imputed_val = svr_model.predict(X_test)[0]
                    
                    # 更新到 data_imputed
                    sample_df.iloc[t, d] = imputed_val
                    series_prefilled[t] = imputed_val # 更新 prefilled 数据，用于后续预测
        
        # 将处理后的数据存回
        data_imputed[i] = sample_df.values
    
    return data_imputed


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


    dm.inject_errors(0.1, ["missing"], covered_attrs=range(data.shape[-1]))

    dirty_data, error_mask = dm.get_dirty_data_restored()













    # data_missing = dm.restored_observed_data
    # missing_mask = data_missing.isna()
    # missing_rate = missing_mask.mean().to_dict()
    # print("missing_rate:", missing_rate)

    # print("修复结果:")
    # imputed_ts = impute_timestamps(data_missing)
    # data_imputed = impute_svr_univariate(data_missing)
    # data_imputed.iloc[:,0] = imputed_ts
    # missing_mask = data_imputed.isna()
    # missing_rate = missing_mask.mean().to_dict()
    # print("missing_rate:", missing_rate)

    
    # # # 查找data_missing中timestamp列缺失值的索引
    # # timestamp_missing_indices = data_missing[data_missing['timestamp'].isna()].index
    # # print("\ntimestamp列缺失值的索引:")
    # # print(timestamp_missing_indices)
    
    # # # 检查data_imputed中对应位置的值
    # # print("\ndata_imputed中对应索引的timestamp值:")
    # # timestamp_imputed_values = data_imputed.loc[timestamp_missing_indices, 'timestamp']
    # # print(timestamp_imputed_values)


    # methods = [
    #     # impute_interpolation,
    #     # impute_moving_average,
    #     impute_ar,
    #     # impute_kalman_filter,
    #     # impute_gmm_em,
    #     # impute_hmm,
    #     # impute_svr_univariate,

    # ]

    # imputed_ts = impute_timestamps(data_missing)
    # results = {}
    # for func in methods:
    #     print(f"\n 运行 {func.__name__} ...")
    #     try:
    #         data_imputed = func(data_missing)
    #         data_imputed.iloc[:,0] = imputed_ts
    #         if not isinstance(data_imputed, pd.DataFrame):
    #             raise ValueError("返回类型错误，必须是 DataFrame")
    #         # 简单的效果验证：
    #         missing_mask = data_imputed.isna()
    #         missing_rate = missing_mask.mean().to_dict()
    #         print("缺失率:", missing_rate)
    #     except Exception as e:
    #         print(f"{func.__name__} 失败: {e}")

