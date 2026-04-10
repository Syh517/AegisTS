import numpy as np

class ZScore:
    def __init__(self):
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        zscore = np.abs((data - mean) / (std + 1e-8))
        self.decision_scores_ = np.max(zscore, axis=1)
        return self
