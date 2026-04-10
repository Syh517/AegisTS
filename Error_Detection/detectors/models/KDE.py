import numpy as np
from sklearn.neighbors import KernelDensity

class KDE:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
        kde.fit(data)
        log_probs = kde.score_samples(data)
        self.decision_scores_ = -log_probs
        return self
