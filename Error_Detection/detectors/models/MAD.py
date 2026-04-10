import numpy as np

class MAD:
    def __init__(self):
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0) + 1e-8
        score = np.abs(data - median) / mad
        self.decision_scores_ = np.max(score, axis=1)
        return self
