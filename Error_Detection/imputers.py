import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import KNNImputer
from pykalman import KalmanFilter


class TimeSeriesImputer:
    def __init__(self, data, method="linear", **kwargs):
        """
        初始化时间序列插补器
        
        Parameters:
        -----------
        data : pandas.DataFrame or numpy.ndarray
            输入的时序数据
        method : str
            插补方法
        **kwargs : dict
            其他参数
        """
        # 保留输入数据副本，防止外部数据被修改
        if isinstance(data, pd.DataFrame):
            self.data = data.copy(deep=True)
        else:  # numpy array
            self.data = np.copy(data)
            
        self.method = method
        self.kwargs = kwargs
        self.is_dataframe = isinstance(data, pd.DataFrame)

    def fit_transform(self):
        """
        执行插补操作
        
        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            插补后的数据
        """
        # 在副本上操作，保证 self.df 不被修改
        if self.is_dataframe:
            data = self.data.copy(deep=True)
        else:
            data = self.data.copy()

        if self.method == "linear":
            if self.is_dataframe:
                return data.interpolate(method="linear")
            else:
                return self._numpy_linear_impute(data)

        elif self.method == "rolling_mean":
            window = self.kwargs.get("window", 10)
            if self.is_dataframe:
                return data.fillna(data.rolling(window=window, min_periods=1).mean())
            else:
                return self._numpy_rolling_mean_impute(data, window)

        elif self.method == "ewma":
            span = self.kwargs.get("span", 3)
            if self.is_dataframe:
                return data.fillna(data.ewm(span=span, adjust=False).mean())
            else:
                return self._numpy_ewma_impute(data, span)

        elif self.method == "knn":
            n_neighbors = self.kwargs.get("n_neighbors", 5)
            if self.is_dataframe:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed = imputer.fit_transform(data)
                return pd.DataFrame(imputed, columns=data.columns, index=data.index)
            else:
                return self._numpy_knn_impute(data, n_neighbors)

        elif self.method == "ridge":
            if self.is_dataframe:
                return self._ridge_impute(data)
            else:
                return self._numpy_ridge_impute(data)

        elif self.method == "kalman":
            if self.is_dataframe:
                return self._kalman_impute(data)
            else:
                return self._numpy_kalman_impute(data)

        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

    def _numpy_linear_impute(self, data: np.ndarray) -> np.ndarray:
        """基于线性插值的numpy数组插补"""
        imputed_data = np.copy(data)
        n_cols = imputed_data.shape[1] if len(imputed_data.shape) > 1 else 1
        
        if len(imputed_data.shape) == 1:
            imputed_data = self._interpolate_1d(imputed_data)
        else:
            for col in range(n_cols):
                imputed_data[:, col] = self._interpolate_1d(imputed_data[:, col])
                
        return imputed_data
    
    def _interpolate_1d(self, arr: np.ndarray) -> np.ndarray:
        """对一维数组进行线性插值"""
        nan_mask = np.isnan(arr)
        if not np.any(nan_mask):
            return arr
            
        indices = np.arange(len(arr))
        valid_indices = indices[~nan_mask]
        valid_values = arr[~nan_mask]

        # All values are NaN; fall back to zeros to avoid np.interp failure.
        if valid_indices.size == 0:
            return np.zeros_like(arr, dtype=float)
        
        # 线性插值
        interpolated = np.interp(indices, valid_indices, valid_values)
        return interpolated

    def _numpy_rolling_mean_impute(self, data: np.ndarray, window: int) -> np.ndarray:
        """基于滚动均值的numpy数组插补"""
        imputed_data = np.copy(data)
        n_cols = imputed_data.shape[1] if len(imputed_data.shape) > 1 else 1
        
        if len(imputed_data.shape) == 1:
            imputed_data = self._rolling_mean_1d(imputed_data, window)
        else:
            for col in range(n_cols):
                imputed_data[:, col] = self._rolling_mean_1d(imputed_data[:, col], window)
                
        return imputed_data
    
    def _rolling_mean_1d(self, arr: np.ndarray, window: int) -> np.ndarray:
        """对一维数组进行滚动均值插补"""
        # 创建滚动窗口
        result = np.copy(arr)
        nan_mask = np.isnan(arr)
        
        if not np.any(nan_mask):
            return result
            
        # 计算滚动均值并填充NaN值
        for i in range(len(arr)):
            if nan_mask[i]:
                start = max(0, i - window + 1)
                end = i + 1
                window_data = arr[start:end]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) > 0:
                    result[i] = np.mean(valid_data)
                    
        return result

    def _numpy_ewma_impute(self, data: np.ndarray, span: int) -> np.ndarray:
        """基于指数加权移动平均的numpy数组插补"""
        imputed_data = np.copy(data)
        n_cols = imputed_data.shape[1] if len(imputed_data.shape) > 1 else 1
        
        if len(imputed_data.shape) == 1:
            imputed_data = self._ewma_1d(imputed_data, span)
        else:
            for col in range(n_cols):
                imputed_data[:, col] = self._ewma_1d(imputed_data[:, col], span)
                
        return imputed_data
    
    def _ewma_1d(self, arr: np.ndarray, span: int) -> np.ndarray:
        """对一维数组进行指数加权移动平均插补"""
        result = np.copy(arr)
        nan_mask = np.isnan(arr)
        
        if not np.any(nan_mask):
            return result
            
        alpha = 2.0 / (span + 1)
        for i in range(1, len(arr)):
            if np.isnan(result[i]):
                # 使用之前的非NaN值进行估计
                result[i] = result[i-1] + alpha * (result[i-1] - result[i-1])  # 简化处理
            else:
                # 更新EWMA值
                result[i] = result[i-1] + alpha * (arr[i] - result[i-1]) if i > 0 else arr[i]
                
        # 向后填充剩余的NaN值
        for i in range(len(arr)-2, -1, -1):
            if np.isnan(result[i]) and not np.isnan(result[i+1]):
                result[i] = result[i+1]
                
        return result

    def _numpy_knn_impute(self, data: np.ndarray, n_neighbors: int) -> np.ndarray:
        """基于KNN的numpy数组插补"""
        # 将NaN值替换为列均值，以便KNN可以处理
        data_filled = np.copy(data)
        n_cols = data_filled.shape[1]
        
        for col in range(n_cols):
            col_data = data_filled[:, col]
            if np.isnan(col_data).any():
                mean_val = np.nanmean(col_data)
                data_filled[:, col] = np.where(np.isnan(col_data), mean_val, col_data)
        
        # 应用KNN插补
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed = imputer.fit_transform(data_filled)
        return imputed

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

    def _numpy_ridge_impute(self, data: np.ndarray) -> np.ndarray:
        """基于回归模型的numpy数组多变量插补"""
        imputed_data = np.copy(data)
        n_cols = imputed_data.shape[1]
        
        for col in range(n_cols):
            col_data = imputed_data[:, col]
            nan_mask = np.isnan(col_data)
            
            if not np.any(nan_mask):
                continue
                
            # 构造特征矩阵X（排除当前列）
            X = np.delete(imputed_data, col, axis=1)
            
            # 获取非NaN值用于训练
            valid_mask = ~nan_mask
            if np.sum(valid_mask) == 0:
                continue
                
            X_train = X[valid_mask]
            y_train = col_data[valid_mask]
            
            # 训练模型
            model = HistGradientBoostingRegressor(random_state=0)
            model.fit(X_train, y_train)
            
            # 预测缺失值
            X_test = X[nan_mask]
            if len(X_test) > 0:
                imputed_data[nan_mask, col] = model.predict(X_test)
                
        return imputed_data

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

    def _numpy_kalman_impute(self, data: np.ndarray) -> np.ndarray:
        """基于卡尔曼滤波的numpy数组逐列插补"""
        imputed_data = np.copy(data)
        n_cols = imputed_data.shape[1] if len(imputed_data.shape) > 1 else 1
        
        if len(imputed_data.shape) == 1:
            try:
                # Reshape for Kalman filter
                data_reshaped = imputed_data.reshape(-1, 1)
                kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                kf = kf.em(data_reshaped, n_iter=5)
                state_means, _ = kf.smooth(data_reshaped)
                imputed_data = state_means.flatten()
            except Exception:
                pass  # Return original data if Kalman fails
        else:
            for col in range(n_cols):
                try:
                    col_data = imputed_data[:, col]
                    # Reshape for Kalman filter
                    data_reshaped = col_data.reshape(-1, 1)
                    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                    kf = kf.em(data_reshaped, n_iter=5)
                    state_means, _ = kf.smooth(data_reshaped)
                    imputed_data[:, col] = state_means.flatten()
                except Exception:
                    pass  # Keep original data if Kalman fails
                    
        return imputed_data