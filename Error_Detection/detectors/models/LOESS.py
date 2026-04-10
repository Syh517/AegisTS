import numpy as np
import statsmodels.api as sm

class LOESS:
    def __init__(self, frac=0.3):
        self.frac = frac
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        smooth = np.zeros_like(data)

        x = np.arange(n_samples)
        for i in range(n_features):
            y = data[:, i]
            # lowess 返回 Nx2 的 array，第二列是平滑值
            smooth[:, i] = sm.nonparametric.lowess(y, x, frac=self.frac, return_sorted=False)

        residual = np.abs(data - smooth)
        self.decision_scores_ = np.max(residual, axis=1)
        return self
