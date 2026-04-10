import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from pykalman import KalmanFilter


class TimeSeriesImputer:
    def __init__(self, df: pd.DataFrame, method="linear", **kwargs):
        # 保留输入数据副本，防止外部数据被修改
        self.df = df.copy(deep=True)
        self.method = method
        self.kwargs = kwargs

    def fit_transform(self) -> pd.DataFrame:
        # 在副本上操作，保证 self.df 不被修改
        data = self.df.copy(deep=True)

        if self.method == "linear":
            return data.interpolate(method="linear")

        elif self.method == "rolling_mean":
            window = self.kwargs.get("window", 10)
            return data.fillna(data.rolling(window=window, min_periods=1).mean())

        elif self.method == "ewma":
            span = self.kwargs.get("span", 3)
            return data.fillna(data.ewm(span=span, adjust=False).mean())

        elif self.method == "knn":
            n_neighbors = self.kwargs.get("n_neighbors", 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = imputer.fit_transform(data)
            return pd.DataFrame(imputed, columns=data.columns, index=data.index)

        elif self.method == "ridge":
            return self._ridge_impute(data)

        elif self.method == "kalman":
            return self._kalman_impute(data)

        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

    def _ridge_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于回归模型的多变量插补（安全版，不修改输入）"""
        df_imputed = data.copy(deep=True)
        for col in data.columns:
            if data[col].isna().sum() == 0:
                continue

            X = data.drop(columns=[col])
            y = data[col]
            mask = y.notna()

            model = HistGradientBoostingRegressor(random_state=0)
            model.fit(X[mask], y[mask])

            df_imputed.loc[~mask, col] = model.predict(X[~mask])
        return df_imputed

    def _kalman_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """基于卡尔曼滤波的逐列插补（安全版，不修改输入）"""
        df_imputed = pd.DataFrame(index=data.index)
        for col in data.columns:
            try:
                kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                kf = kf.em(data[col], n_iter=5)
                state_means, _ = kf.smooth(data[col])
                df_imputed[col] = state_means.flatten()
            except Exception:
                df_imputed[col] = data[col]
        return df_imputed
