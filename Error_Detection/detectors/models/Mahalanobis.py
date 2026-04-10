import numpy as np

class Mahalanobis:
    def __init__(self):
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        diff = data - mean
        dists = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        self.decision_scores_ = dists
        return self
