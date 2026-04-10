import numpy as np

class IQR:
    def __init__(self, k=1.5):
        self.k = k
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        lower = q1 - self.k * (q3 - q1)
        upper = q3 + self.k * (q3 - q1)
        outliers = (data < lower) | (data > upper)
        self.decision_scores_ = np.max(outliers.astype(float), axis=1)
        return self
