import numpy as np
import pandas as pd
import os
import sys
try:
    from ts2vec import TS2Vec
except Exception:
    TS2Vec = None

try:
    import hdbscan
except Exception:
    hdbscan = None
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import random
from collections import defaultdict, Counter


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Datasets.load_dataset import load_single_dataset, sample_data_by_rate, convert_to_unix_timestamp

from Datasets.load_dataset import load_single_dataset, sample_data_by_rate
from Error_Injection.injector import DataManager
from Error_Detection.imputers import TimeSeriesImputer
from Error_Detection.Outliers import OutlierDetector
from Error_Detection.ts_column import clean_timestamp_column_with_regression

# from Error_Detection.miner import mine_all_constraints,constraint_report
# from Error_Detection.Constraints import ConstraintViolationDetector

# from Error_Detection.miner_2type_row_liner_col import mine_all_constraints, constraint_report, mine_all_constraints_per_sample
from Error_Detection.miner_2type_row_poly_col import mine_all_constraints, constraint_report, mine_all_constraints_per_sample
from Error_Detection.Constraints_2type import ConstraintViolationDetector


def load_original_data(dataset_name, dataset_type, rate=1):
    original_dataset_dir = "/home/syh/TSC/AutoClean/Datasets/original"
    if dataset_type == 'clean':
        csv_file = f"{original_dataset_dir}/{dataset_name}/{dataset_name}_Clean.csv"
    elif dataset_type == 'dirty':
        csv_file = f"{original_dataset_dir}/{dataset_name}/{dataset_name}_Dirty.csv"

    try:
        df = pd.read_csv(csv_file)
        
        # 根据rate获取前rate%的数据
        if rate < 1:
            n_rows = len(df)
            n_selected = max(1, int(n_rows * rate))
            # 对于时间序列数据，我们使用前n_selected行以保持时间顺序
            df = df.iloc[:n_selected]
        
        # 检查并转换时间列
        for col in df.columns:
            # 检查是否是时间相关的列（第一列通常为时间戳）
            if col.lower() in ['timestamp', 'time', 'date', 'datetime']:
                df[col] = convert_to_unix_timestamp(df[col])
            # 即使列名不明显是时间列，也尝试检测是否包含时间字符串
            else:
                # 取样前几个非空值进行检测
                sample_values = df[col].dropna().head(5)
                if len(sample_values) > 0 and all(isinstance(val, str) for val in sample_values):
                    # 检查是否可能是日期时间格式
                    try:
                        pd.to_datetime(sample_values, errors='raise')
                        df[col] = convert_to_unix_timestamp(df[col])
                    except (ValueError, TypeError, OverflowError):
                        # 不是时间格式，跳过
                        pass
        
        data = df.values
        
        # 转换为(1, n_timestamps, n_features)格式
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        return data
    
    except Exception as e:
        raise RuntimeError(f"加载 {csv_file} 时出错: {e}")


class Detector:
    def __init__(self, data_manager, miner_type=True):
        """
        初始化检测器。
        :param data_manager: 一个 DataManager 实例。
        """
        self.dm = data_manager
        self.miner_type = miner_type
        self.error_detect = None
        self.missing = []
        self.duplicate = []


    def _get_data(self):
        # drop timestamp
        if isinstance(self.data, np.ndarray) and len(self.data.shape) == 3:
            # 三维数组 (n_samples, n_timestamps, n_features)
            # 返回所有样本的特征数据，去掉第一列(timestamp)
            data = self.data[:, :, 1:].astype(float).copy()
            return data


    def detect_missing(self):
        """
        检测缺失值和重复值，更新 self.error_detect。
        """
            
        self.missing = []
        for sample_idx in range(self.data.shape[0]):
            # Handle mixed data types (string and numeric)
            sample_data = self.data[sample_idx]
            missing_mask = np.zeros(sample_data.shape, dtype=bool)
            
            # Check each column for missing values
            for col in range(sample_data.shape[1]):
                col_data = sample_data[:, col]
                # Create a series to handle mixed types properly
                col_series = pd.Series(col_data)
                
                # Use pandas' isna function which handles multiple data types
                missing_mask[:, col] = col_series.isna() | (col_series == '') | (col_series.str.lower() == 'nan') if col_series.dtype == 'object' else np.isnan(col_series.astype(float, copy=False))
            
            self.missing.append(missing_mask.any())
            self.error_detect[sample_idx, :, :] = missing_mask
            
        return self.missing

    def detect_duplicate(self, min_repeat_len=8, check=False):
        """
        检测重复值异常：识别整行与前一行完全相同的连续段，并将重复段中的第二行及后续行设为NaN。

        参数：
            min_repeat_len (int): 最小重复段长度（含首行）。
        """
            
        self.duplicate = []
        
        # 处理numpy数组

        if check == False:
            for sample_idx in range(self.data.shape[0]):
                sample_duplicate = self._detect_duplicate_2d(sample_idx, min_repeat_len)
                self.duplicate.append(sample_duplicate)
        else:
            for sample_idx in range(self.data.shape[0]):
                sample_duplicate = self._detect_duplicate_2d_check(sample_idx, min_repeat_len)
                self.duplicate.append(sample_duplicate)

                        
        return self.duplicate

    def _detect_duplicate_2d(self, sample_idx, min_repeat_len=8):
        """
        在二维数组上检测重复值
        """
        duplicate_found = False
        data_2d = self.data[sample_idx]  # (n_timestamps, n_features)
        if data_2d.shape[1] >= 3:
            repeated = np.all(data_2d == np.roll(data_2d, 1, axis=0), axis=1)
            repeated[0] = False  # 第一行无前驱，不可能重复

            n = len(data_2d)
            i = 0
            while i < n:
                if not repeated[i]:
                    i += 1
                    continue

                # 找到从 i 开始的重复段 [i, j)
                j = i + 1
                while j < n and repeated[j]:
                    j += 1

                seg_len = j - i + 1  # 包括最初不在 repeated 中的那一行
                if seg_len >= min_repeat_len:
                    # 标记重复异常（从 i 到 j-1）
                    self.error_detect[sample_idx, i:j, :] = True
                    self.data[sample_idx, i:j, :] = np.nan
                    duplicate_found = True

                i = j
                
        return duplicate_found

    def _detect_duplicate_2d_check(self, sample_idx, min_repeat_len=8):
        """
        在二维数组上检测重复值
        """
        duplicate_found = False
        data_2d = self.data[sample_idx]  # (n_timestamps, n_features)
        if data_2d.shape[1] >= 3:
            repeated = np.all(data_2d == np.roll(data_2d, 1, axis=0), axis=1)
            repeated[0] = False  # 第一行无前驱，不可能重复

            n = len(data_2d)
            i = 0
            while i < n:
                if not repeated[i]:
                    i += 1
                    continue

                # 找到从 i 开始的重复段 [i, j)
                j = i + 1
                while j < n and repeated[j]:
                    j += 1

                seg_len = j - i + 1  # 包括最初不在 repeated 中的那一行
                if seg_len >= min_repeat_len:
                    duplicate_found = True
                    break

                i = j
                
        return duplicate_found

    def get_missing_duplicate_metrics(self, min_repeat_len=8):
        # 检测缺失值并不让error_mask变化，只检测缺失值是否存在
        self.missing = []
        for sample_idx in range(self.data.shape[0]):
            # Handle mixed data types (string and numeric)
            sample_data = self.data[sample_idx]
            missing_mask = np.zeros(sample_data.shape, dtype=bool)
            
            # Check each column for missing values
            for col in range(sample_data.shape[1]):
                col_data = sample_data[:, col]
                # Create a series to handle mixed types properly
                col_series = pd.Series(col_data)

                # if isinstance(col_data[0], str):
                #     missing_mask[:, col] = (col_data == 'nan') | (col_data == 'NaN') | (col_data == '') | pd.isna(col_data)
                # else:
                #     missing_mask[:, col] = np.isnan(col_data.astype(float))
                
                # Use pandas' isna function which handles multiple data types
                missing_mask[:, col] = col_series.isna() | (col_series == '') | (col_series.str.lower() == 'nan') if col_series.dtype == 'object' else np.isnan(col_series.astype(float, copy=False))
            
            self.missing.append(missing_mask.any())

        
        # 检测重复值
        self.duplicate = self.detect_duplicate(min_repeat_len=min_repeat_len, check=True)

    # def get_missing_score(self):
    #     missing_rates = []
    #     for sample_idx in range(self.data.shape[0]):
    #         sample_data = self.data[sample_idx]
    #         # Convert to float array (will turn non-numeric like 'nan', '' into NaN)
    #         sample_float = sample_data.astype(float)
    #         # Use np.isnan to detect all NaN values
    #         missing_count = np.count_nonzero(np.isnan(sample_float))
    #         missing_rate = missing_count / sample_data.size
    #         missing_rates.append(missing_rate)
    #     return missing_rates

    
    def get_missing_score(self):
        missing_rates = []
        for sample_idx in range(self.data.shape[0]):
            # Handle mixed data types (string and numeric)
            sample_data = self.data[sample_idx]
            missing_mask = np.zeros(sample_data.shape, dtype=bool)
            
            # Check each column for missing values
            for col in range(sample_data.shape[1]):
                col_data = sample_data[:, col]
                # Create a series to handle mixed types properly
                col_series = pd.Series(col_data)
                
                # Use pandas' isna function which handles multiple data types
                missing_mask[:, col] = col_series.isna() | (col_series == '') | (col_series.str.lower() == 'nan') if col_series.dtype == 'object' else np.isnan(col_series.astype(float, copy=False))
            
            missing_rates.append(np.mean(missing_mask))
        return missing_rates
    
        
    def impute_timestamps(self, data_2d):

        # 对于numpy数组，假设第一列是timestamp
        timestamps = data_2d[:, 0]
        if len(timestamps) == 0:
            return np.array([])

        # 检查是否所有非空值都是字符串
        non_na_vals = [x for x in timestamps if pd.notna(x)]
        is_all_str = non_na_vals and all(isinstance(x, str) for x in non_na_vals)

        is_datetime = False
        if is_all_str:
            try:
                pd.to_datetime(non_na_vals, errors="raise")
                is_datetime = True
            except (ValueError, TypeError, OverflowError):
                is_datetime = False

        # print(f"is_datetime = {is_datetime}")

        if is_datetime:
            ts_series = pd.Series(pd.to_datetime(timestamps))
            # 使用线性插值（基于索引），因为 'time' 方法可能不可用
            ts_interp = ts_series.interpolate(method='linear')
            return ts_interp.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

        # 否则当作数值处理
        try:
            numeric_series = pd.to_numeric(timestamps, errors="coerce")
            numeric_interp = pd.Series(numeric_series).interpolate(method='linear')
            return numeric_interp.to_numpy()
        except Exception:
            # 兜底
            return pd.Series(timestamps).interpolate(method='linear').to_numpy()


    def preprocess_sample(self, sample_idx):
        """
        对单个样本进行预处理
        :param sample_data: 单个样本的二维数据 (n_timestamps, n_features)
        """
        # print("---Preprocess Missing and Duplicate Errors for a sample...---")
        
        data_2d = np.copy(self.data[sample_idx])  # (n_timestamps, n_features)
    
        
        if self.duplicate[sample_idx] or self.missing[sample_idx]:
            # print("--->Need Imputation for sample")
            # 进行插补
            # imputed_timestamp = self.impute_timestamps(data_2d)

            # variables_2_imputed = data_2d[:, 1:].astype(float)
            # imputer1 = TimeSeriesImputer(variables_2_imputed, method="rolling_mean")
            # variables_2_imputed = imputer1.fit_transform()
            # imputer2 = TimeSeriesImputer(variables_2_imputed, method="linear")
            # imputed_variables = imputer2.fit_transform()

            # data_imputed = np.column_stack((imputed_timestamp, imputed_variables))

            # imputer1 = TimeSeriesImputer(data_2d, method="rolling_mean")
            # data_imputed = imputer1.fit_transform()
            # imputer2 = TimeSeriesImputer(data_2d, method="linear")
            # data_imputed = imputer2.fit_transform()

            imputer = TimeSeriesImputer(data_2d, method="linear")
            data_imputed = imputer.fit_transform()

            # print("sample_shape:", data_imputed.shape)
            return data_imputed  # 返回numpy数组
        else:
            # print("--->No Imputation needed for sample")
            return data_2d

    def preprocess(self):
        print("---Preprocess Missing and Duplicate Errors...---")

        # 对整个数据集进行检测
        if self.flag == False:
            print("self.missing:",  any(self.detect_missing()))
            print("self.duplicate:", any(self.detect_duplicate()))
            # print("self.duplicate:", self.duplicate)
        else:
            print("self.missing:",  any(self.detect_missing()))
            print("self.duplicate:", any(self.detect_duplicate(check=True)))


        self.dirty_data_w_missing = clean_timestamp_column_with_regression(self.data)
        self.data = self.dirty_data_w_missing.copy()
        missing_rates = self.get_missing_score()

        if any(self.duplicate) or any(self.missing):
            print("--->Need Imputation")

            
            # 三维数组 (n_samples, n_timestamps, n_features)
            # 对每个样本分别处理
            processed_samples = []
            for sample_idx in range(self.data.shape[0]):
                processed_sample = self.preprocess_sample(sample_idx)
                processed_samples.append(processed_sample)
            
            # 合并处理后的样本
            self.data = np.stack(processed_samples, axis=0)

            # 检验插补后的缺失重复情况
            # self.get_missing_duplicate_metrics()
            print("self.missing:",  any(self.detect_missing()))
            print("self.duplicate:", any(self.detect_duplicate(check=True)))
            # print("self.duplicate:", self.detect_duplicate())
        else:
            print("--->No Imputation")

        print("---Preprocess finished---\n")

        return missing_rates


    def detect_outlier(self, data, n_selection=5, threshold=0.5):
        # 选择合适的异常检测方法，不可能所有检测方法都用上
        outlier_detector = OutlierDetector(n_selection)
        

        n_samples, n_timestamps, n_features = data.shape
        
        models=[]
        for i in range(min(n_samples, 3)):
            random_sample_idx = np.random.choice(n_samples, 1, replace=False)
            random_sample = data[random_sample_idx].reshape(n_timestamps, n_features)
            random_sample = random_sample[:,1:] #去掉第一列时间列
            # print("Random sample shape for model selection:", random_sample.shape)
            selected_models = outlier_detector.get_admodels(random_sample)
            # print("Selected models: ", selected_models)
            models.extend(selected_models.tolist() if hasattr(selected_models, 'tolist') else selected_models)
        model_counter = Counter(models)
        top_n_model_names = [name for name, _ in model_counter.most_common(3)]
        # print("Final selected models: ", top_n_model_names )

        outlier_rates = []
        outlier_predict_labels = []
        for sample_idx in range(n_samples):
            sample_data = data[sample_idx].reshape(n_timestamps, n_features)
            sample_data = sample_data[:,1:] #去掉第一列时间列
            sample_ad_scores = outlier_detector.get_adscores(top_n_model_names, sample_data)  
            threshold = np.mean(sample_ad_scores) + 3 * np.std(sample_ad_scores)  # 利用阈值筛选
            sample_outlier_predict = (sample_ad_scores > threshold)  #1D array
            outlier_predict_labels.append(sample_outlier_predict)
            sample_outlier_rate = sample_outlier_predict.mean()  #异常率 
            outlier_rates.append(sample_outlier_rate)

        if len(outlier_rates) == 0:
            outlier_rate_avg = 0.0  
        else:
            outlier_rate_avg = np.array(outlier_rates).mean()

        # print("Outlier scores:", outlier_rate_avg)

        return outlier_predict_labels, outlier_rates, outlier_rate_avg


    def detect_constraint_violation(self, data):
        # 创建约束违反检测器
        print("检测约束违反...")
        
        Cdetector = ConstraintViolationDetector(self.constraints)
                
        
        # 检测所有违反
        violations_3d = Cdetector.detect_all_violations(data)

        # 统计所有违反
        sample_total_constraint_violation_rates = {}  # Initialize with default value
        try:
            # violation_rates_3d = Cdetector.calculate_violation_rates(data)
            # print("\n3D数据违反率统计:")
            # print(violation_rates_3d)
            
            # 计算每个样本的违反率
            sample_violation_rates = Cdetector.calculate_sample_violation_rates(data)
            # 按指定格式输出
            sample_total_constraint_violation_rates = Cdetector.print_sample_violation_rates(sample_violation_rates)
        except Exception as e:
            print(f"计算违反率时出错: {e}")
            # Return empty list or appropriate default value when calculation fails
            sample_total_constraint_violation_rates = {}

        
        return sample_total_constraint_violation_rates
    

    def detect_all(self, data_to_detect=None):
        print("-----Detect Errors...-----")

        if data_to_detect is not None:
            self.data = data_to_detect.copy()
            self.flag = True
        else:
            self.data = self.dm.observed_data_restored.copy()
            self.flag = False

        print("data shape:", self.data.shape)
        
        # Initialize error_detect with correct shape
        if isinstance(self.data, np.ndarray) and len(self.data.shape) == 3:
            self.error_detect = np.full(self.data.shape, False, dtype=bool)

        missing_rates = self.preprocess()
        # self.data = clean_timestamp_column_with_regression(self.data)

        if self.flag == False:
            outlier_predict_labels, outlier_rates, outlier_rate_avg = self.detect_outlier(self.data.copy(), n_selection=5, threshold=0.5)

            
            # 挖掘约束
            print("挖掘数据约束...")
            
            # 准备用于约束挖掘的数据
            n_samples, n_timestamps, n_features = self.data.shape

            if self.miner_type:  # True: 使用统一的约束条件
                # 选择3个随机样本进行约束挖掘
                if n_samples < 3:
                    random_samples = self.data
                    random_outlier_rate = outlier_rate_avg
                else:
                    random_sample_idx = np.random.choice(n_samples, size=3, replace=False)  # shape: (3,)
                    random_samples = self.data[random_sample_idx]  # 自动得到 shape: (3, n_timestamps, n_features)
                    outlier_rates = np.array(outlier_rates)
                    random_outlier_rate = outlier_rates[random_sample_idx].mean()  # shape 取决于 outlier_rates 的 shape

                self.constraints = mine_all_constraints(data_3d=random_samples, degree=2, attr_num=2, outlier_rate=random_outlier_rate, window=50, min_support=0.1, min_conf=0.9)
                # print("constraints:", self.constraints)
                constraint_report(self.constraints)

                violation_rates = self.detect_constraint_violation(self.data)
            else:
                self.constraints = mine_all_constraints_per_sample(data_3d=self.data, degree=2, attr_num=2, outlier_rate=outlier_rates, window=50, min_support=0.1, min_conf=0.9)

                violation_rates = self.detect_constraint_violation(self.data)


            # self.flag = True    
        else:
            outlier_predict_labels, outlier_rates, outlier_rate_avg = self.detect_outlier(self.data.copy(), n_selection=5, threshold=0.5)
            violation_rates = self.detect_constraint_violation(self.data)

        print("-----Detect finished-----\n\n")

        # print("missing_rates:", np.array(missing_rates))  #每个sample的缺失率
        # print("outlier_rates:", np.array(outlier_rates))  #每个sample的异常率
        # print("outlier_predict_labels:", np.array(outlier_predict_labels).shape) # shape (num_samples, n_timestamps), binary values indicating anomalies 每个sample的异常预测labels
        # print("violation_rates:", violation_rates) #每个sample的约束违反率
        # print("constraints:", self.constraints)

        if self.flag == False:
            return np.array(missing_rates), np.array(outlier_rates), np.array(outlier_predict_labels), violation_rates, self.constraints, self.dirty_data_w_missing
        else:
            return np.array(missing_rates), np.array(outlier_rates), np.array(outlier_predict_labels), violation_rates, self.constraints



if __name__ == "__main__":

    type = 'forecast'
    dataset_name = 'ETTh1'
    data, label_2_train =load_single_dataset(type, dataset_name, rate=0.1)

    # data = load_original_data('IDF_OilTemp', 'clean', rate=1)
    # label_2_train = None
    
    # type = 'classification'
    # dataset_name = 'Libras'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
    # data, label_2_train = sample_data_by_rate(data, label_2_train, rate=0.5)

    # type = 'classification'
    # dataset_name = 'Handwriting'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=1)
    # data, label_2_train = sample_data_by_rate(data, label_2_train, rate=0.2)

    if type == 'forecast':
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        print(data.shape)
        dm = DataManager(data, abnormal_rate=0.3, task_type=type)
    else:
        print(data.shape, label_2_train.shape)
        dm = DataManager(data, label_2_train, abnormal_rate=0.3, task_type=type)




    # 注入错误
    dm.inject_errors(
        0.3,
        ["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
        covered_attrs=range(data.shape[-1]),
    )

    


    # data_dirty, error_mask = dm.get_dirty_data_restored()
    # print("data_dirty:", data_dirty[0,:10,:])
    

    # dm.inject_errors(
    #     0.1,
    #     ["duplicate", "single"],
    #     covered_attrs=range(data.shape[-1]),
    # )




    # dm.inject_errors(
    #     0.1,
    #     ["single", "drift", "gaussian", "volatility", "gradual", "sudden"],
    #     covered_attrs=range(data.shape[-1]),
    # )

    # dm.inject_errors(
    #     0.1,
    #     [
    #         "single",
    #         "drift",
    #         "gaussian",
    #         "volatility",
    #         "gradual",
    #         "sudden",
    #         "missing",
    #         "duplicate",
    #     ],
    #     covered_attrs = range(data.shape[-1]) 
    # )

    # print(dm.clean_data_raw.shape)
    # print(dm.observed_data_restored.shape)    
    # print(dm.error_mask_restored.shape)

    # 异常检测
    miner_type = True # True: 使用统一的约束条件, False: 每个sample使用不同的约束条件
    detector = Detector(dm, miner_type=miner_type)
    missing_rate, outlier_rate, outlier_predict_labels, violation_rates, constraints, data_2_repair = detector.detect_all()
    # print("constraints:", constraints['row_constraints'])
    print("violation_rates", violation_rates)


