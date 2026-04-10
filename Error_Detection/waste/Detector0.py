import numpy as np
import pandas as pd
import os
import sys
from ts2vec import TS2Vec
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import random
from collections import defaultdict


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Error_Injection.injector import DataManager
from Error_Detection.imputers import TimeSeriesImputer
from Error_Detection.Outliers import OutlierDetector

from Error_Detection.miner0 import mine_all_constraints,constraint_report
from Error_Detection.Constraints import ConstraintViolationDetector, get_violation_rates, print_violation_rates



class Detector:
    def __init__(self, data_manager):
        """
        初始化检测器。
        :param data_manager: 一个 DataManager 实例。
        """
        self.dm = data_manager
        self.error_detect = pd.DataFrame(
            False, index=self.dm.restored_observed_data.index, columns=self.dm.restored_observed_data.columns
        )

        self.missing = False
        self.duplicate = False


    def _get_data(self):
        # drop timestamp
        return self.data.iloc[:, 1:]

    def detect_missing(self):
        """
        检测缺失值和重复值，更新 self.error_detect。
        """
        self.missing = False
        # 检测缺失值
        missing_mask = self.data.isna()
        self.missing = missing_mask.values.any()
        self.error_detect = missing_mask
        return self.missing

    def detect_duplicate(self, min_repeat_len=8):
        """
        检测重复值异常：识别整行与前一行完全相同的连续段，并将重复段中的第二行及后续行设为NaN。

        参数：
            min_repeat_len (int): 最小重复段长度（含首行）。
        """
        self.duplicate = False
        if len(self.data.columns) >= 3:
            repeated = (self.data == self.data.shift(1)).all(axis=1)
            repeated.iloc[0] = False  # 第一行无前驱，不可能重复

            n = len(self.data)
            i = 0
            while i < n:
                if not repeated.iloc[i]:
                    i += 1
                    continue

                # 找到从 i 开始的重复段 [i, j)
                j = i + 1
                while j < n and repeated.iloc[j]:
                    j += 1

                seg_len = j - i + 1  # 包括最初不在 repeated 中的那一行
                if seg_len >= min_repeat_len:
                    # 标记重复异常（从 i 到 j-1），并将重复段的第二行及之后行设为 NaN
                    self.error_detect.iloc[i:j, :] = True
                    self.data.iloc[i:j, :] = np.nan
                    self.duplicate = True

                i = j

        return self.duplicate

    def get_missing_duplicate_metrics(self, min_repeat_len=8):
        # 检测缺失值
        missing_mask = self.data.isna()
        self.missing = missing_mask.values.any()

        # 检测重复值
        self.duplicate = False
        if len(self.data.columns) >= 3:
            repeated = (self.data == self.data.shift(1)).all(axis=1)
            repeated.iloc[0] = False  # 第一行无前驱，不可能重复

            n = len(self.data)
            i = 0
            while i < n:
                if not repeated.iloc[i]:
                    i += 1
                    continue

                # 找到从 i 开始的重复段 [i, j)
                j = i + 1
                while j < n and repeated.iloc[j]:
                    j += 1

                seg_len = j - i + 1  # 包括最初不在 repeated 中的那一行
                if seg_len >= min_repeat_len:
                    self.duplicate = True
                    break

                i = j

    def get_missing_score(self):
        missing_mask = self.data.isna()
        missing_rate = missing_mask.mean().to_dict()
        return missing_rate
    
    def impute_timestamps(self):
        timestamps = self.data.iloc[:, 0]

        if timestamps.isna().any():
            # 判断是否可以转换为 datetime 类型
            try:
                ts = pd.to_datetime(timestamps, errors="raise")
                imputed_ts = ts.interpolate(method="time")
            except Exception:
                imputed_ts = timestamps.interpolate(method="linear")
        else:
            imputed_ts = timestamps

            # data_missing.iloc[:, 0] = imputed_ts

        return imputed_ts
    

    def preprocess(self):
        print("---Preprocess Missing and Duplicate Errors...---")

        print("self.missing:", self.detect_missing())
        print("self.duplicate:", self.detect_duplicate())
        
        missing_rate = self.get_missing_score()

        if self.duplicate == True or self.missing == True:
            # 选择一个合适的方法进行插补，线性插补挺好的
            print("--->Need Impution")

            imputed_timestamp = self.impute_timestamps()
            imputer = TimeSeriesImputer(self.data.copy(deep=True),method="linear")
            data_imputed = imputer.fit_transform()
            data_imputed.iloc[:,0] = imputed_timestamp
            self.data = data_imputed


            # timestamps = self.data.iloc[:, 0] # 第一列 timestamp
            # features = self.data.iloc[:, 1:]

            # imputer = TimeSeriesImputer(method="linear")
            # timestamps_imputed = imputer.impute_timestamps(timestamps)
            # features_imputed = imputer.fit_transform(features)
            # self.data = pd.concat(
            #     [pd.Series(timestamps_imputed, name="timestamp"), pd.DataFrame(features_imputed, columns=features.columns)],
            #     axis=1)

            # 检验插补后的缺失重复情况
            self.get_missing_duplicate_metrics()
            print("self.missing:", self.missing)
            print("self.duplicate:", self.duplicate)
        else:
            print("--->No Impution")

        # self.split()
        # patches, patches_detect_mask, patches_true_mask = self.bucket(self.train_data, self.detect_train_mask, self.true_train_mask)

        print("---Preprocess finished---\n")

        return missing_rate


    def detect_outlier(self, data, n_selection=5, threshold=0.5):
        # 选择合适的异常检测方法，不可能所有检测方法都用上
        outlier_detector = OutlierDetector(n_selection)
        models = outlier_detector.get_admodels(data)

        data_arr = data.to_numpy() # pandas DataFrame 转 numpy array, 因为模型的输入是numpy array
        ad_scores = outlier_detector.get_adscores(models, data_arr) #1D array
        outlier_predict = (ad_scores > 0.5) #1D array
        outlier_rate = outlier_predict.mean() #异常率 flase为正常，true为异常

        # print("Outlier scores:", scores)

        return ad_scores, outlier_predict, outlier_rate


    def detect_constraint_violation(self, data):
                
        # # 挖掘约束
        # print("挖掘数据约束...")
        # attr_num = data.shape[1]
        # self.constraints = mine_all_constraints(df=data, attr_num=attr_num, outlier_rate=outlier_rate, window=50,  min_support=0.1, min_conf=0.9)
        # constraint_report(self.constraints)
        
        
        # 创建约束违反检测器
        print("检测约束违反...")
        Cdetector = ConstraintViolationDetector(self.constraints)
        
        # 检测所有违反
        self.violations = Cdetector.detect_all_violations(data)
        # print("Constraint violations:", self.violations)
        violation_rates = get_violation_rates(self.violations, data)

        constraint_violation_degree = Cdetector.cv
        
        return constraint_violation_degree, violation_rates
    

    def detect_all(self, data_to_detect=None):
        print("-----Detect Errors...-----")

        if data_to_detect is not None:
            self.data = data_to_detect.copy(deep=True)
            self.flag = True
        else:
            self.data = self.dm.restored_observed_data.copy(deep=True)
            self.flag = False

        print("data shape:", self.data.shape)

        missing_rate = self.preprocess()

        if self.flag == False:
            ad_scores, outlier_predict, outlier_rate = self.detect_outlier(self._get_data(), n_selection=5, threshold=0.5)
            
            # 挖掘约束
            print("挖掘数据约束...")
            attr_num = self.data.shape[1]
            self.constraints = mine_all_constraints(df=self.data, attr_num=attr_num, outlier_rate=outlier_rate, window=50,  min_support=0.1, min_conf=0.9)
            print("constraints:", self.constraints)
            constraint_report(self.constraints)

            constraint_violation_degree, violation_rates = self.detect_constraint_violation(self.data)

            self.flag = True
        else:
            ad_scores, outlier_predict, outlier_rate = self.detect_outlier(self._get_data(), n_selection=5, threshold=0.5)
            constraint_violation_degree, violation_rates = self.detect_constraint_violation(self.data)

        # # 综合两种检测结果
        # if ad_scores.shape[0] != constraint_violation_degree.shape[0]:
        #     print("ad_scores shape:", ad_scores.shape)
        #     print("constraint_violation_degree shape:", constraint_violation_degree.shape)
        #     raise ValueError("ad_scores and constraint_violation_degree must have the same shape.")
        # else:
        #     ad_scores_series = pd.Series(ad_scores, index=constraint_violation_degree.index)
        #     weighted_violations = ad_scores_series * constraint_violation_degree

        # print(type(ad_scores))

        print("-----Detect finished-----\n\n")

        print("missing_rate:", missing_rate)
        print("outlier_rate:", outlier_rate)
        print_violation_rates(violation_rates)

        return missing_rate, outlier_rate, violation_rates, self.constraints


def evaluate(pred, true):
    TP = np.logical_and(pred, true).sum()
    TN = np.logical_and(~pred, ~true).sum()
    FP = np.logical_and(pred, ~true).sum()
    FN = np.logical_and(~pred, true).sum()

    print(f"TP: {TP:.4f}")
    print(f"TN: {TN:.4f}")
    print(f"FP: {FP:.4f}")
    print(f"FN: {FN:.4f}")

    accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
    precision = TP / (TP + FP) if (TP + FP) else 0.0  # 精确率 / 查准率
    recall = TP / (TP + FN) if (TP + FN) else 0.0  # 召回率 / 查全率
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    dm = DataManager("idf", "/home/yyy/TSC/TSClean/AutoClean/Datasets/original/IDF_Power/IDF_Power_Clean.csv")
    # dm.inject_errors(0.1, ["missing", "duplicate"], covered_attrs=dm.clean_data.columns)
    # dm.inject_errors(
    #     0.1,
    #     ["single", "drift", "gaussian", "volatility", "gradual", "sudden"],
    #     covered_attrs=dm.clean_data.columns,
    # )
    dm.inject_errors(0.1, ['single', 'drift', 'gaussian', 'volatility', 'gradual', 'sudden', 'missing', 'duplicate'], covered_attrs=dm.clean_data.columns)

    # dm = DataManager("stock","/home/yyy/TSC/TSClean/AutoClean/Datasets/Stock/Stock_Clean.csv","/home/yyy/TSC/TSClean/AutoClean/Datasets/Stock/Stock_Dirty.csv",)

    # 异常检测
    detector = Detector(dm)
    detector.detect_all()

    # pred = detector.detect_train_mask.values  # 检测到的异常 mask
    # true = detector.true_train_mask.values  # 真实异常 mask
    # evaluate(pred, true)

    # # 查看某列检测结果
    # pred = detector.error_detect.values  # 检测到的异常 mask
    # true = dm.error_mask.values  # 真实异常 mask
    # evaluate(pred, true)

    # for col in dm.observed_data.columns:
    #     print("col:",col)
    #     # 查看某列检测结果
    #     pred = dm.error_detect[col].values      # 检测到的异常 mask
    #     true = dm.error_mask[col].values        # 真实异常 mask
    #     evaluate(pred, true)

    # error_detect = dm.error_detect[dm.error_detect[col]].index.tolist()
    # error_mask = dm.error_mask[dm.error_mask[col]].index.tolist()
    # print(f"检测异常的位置 ({col}):", error_detect)
    # print(f"实际异常的位置 ({col}):", error_mask)

