import numpy as np

class EWMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        smooth = np.zeros_like(data)
        smooth[0] = data[0]

        for t in range(1, n_samples):
            smooth[t] = self.alpha * data[t] + (1 - self.alpha) * smooth[t - 1]

        residual = np.abs(data - smooth)
        self.decision_scores_ = residual.max(axis=1)
        return self
