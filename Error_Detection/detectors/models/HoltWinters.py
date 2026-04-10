import numpy as np

class HoltWinters:
    def __init__(self, alpha=0.3, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        level = data[0].copy()
        trend = np.zeros(n_features)
        smooth = np.zeros_like(data)

        for t in range(n_samples):
            smooth[t] = level + trend
            if t < n_samples - 1:
                new_level = self.alpha * data[t+1] + (1 - self.alpha) * (level + trend)
                new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
                level, trend = new_level, new_trend

        residual = np.abs(data - smooth)
        self.decision_scores_ = residual.max(axis=1)
        return self
