import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from pykalman import KalmanFilter
from sklearn.ensemble import HistGradientBoostingRegressor


class TimeSeriesImputer:
    def __init__(self, method="linear", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 线性插值 +
        if self.method == "linear":
            return df.interpolate(method="linear")

        # # 前向填充
        # elif self.method == "ffill":
        #     return df.ffill()

        # # 后向填充
        # elif self.method == "bfill":
        #     return df.bfill()

        # # 均值填充
        # elif self.method == "mean":
        #     return df.fillna(df.mean())

        # 滑动平均填充（支持自定义窗口大小）+
        elif self.method == "rolling_mean":
            window = self.kwargs.get("window", 10)
            return df.fillna(df.rolling(window=window, min_periods=1).mean())

        # 指数加权移动平均（支持自定义平滑系数）
        elif self.method == "ewma":
            span = self.kwargs.get("span", 3)
            return df.fillna(df.ewm(span=span, adjust=False).mean())

        # KNN 插补（支持 n_neighbors 参数）
        elif self.method == "knn":
            n_neighbors = self.kwargs.get("n_neighbors", 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = imputer.fit_transform(df)
            return pd.DataFrame(imputed, columns=df.columns, index=df.index)

        # 基于 Ridge 回归的多变量插补
        elif self.method == "ridge":
            return self._ridge_impute(df)

        # 基于卡尔曼滤波器的逐列插补
        elif self.method == "kalman":
            return self._kalman_impute(df)

        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

    def _ridge_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_imputed = df.copy()
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            X = df.drop(columns=[col])
            y = df[col]

            mask = y.notna()
            model = HistGradientBoostingRegressor(random_state=0)
            model.fit(X[mask], y[mask])

            df_imputed.loc[~mask, col] = model.predict(X[~mask])
        return df_imputed

    def _kalman_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df_imputed = pd.DataFrame(index=df.index)
        for col in df.columns:
            try:
                kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                kf = kf.em(df[col], n_iter=5)
                state_means, _ = kf.smooth(df[col])
                df_imputed[col] = state_means.flatten()
            except:
                df_imputed[col] = df[col]
        return df_imputed
    


    @staticmethod
    def impute_timestamps(timestamps):
        # 判断类型
        try:
            ts = pd.to_datetime(timestamps)
            return ts.interpolate(method="time")
        except Exception:
            # 整数序列或浮点数
            return timestamps.interpolate(method="linear")

