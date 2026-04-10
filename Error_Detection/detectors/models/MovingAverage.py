import numpy as np

class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        smooth = np.empty_like(data)
        half_w = self.window // 2

        # 移动平均（center=True）
        for i in range(n_samples):
            start = max(0, i - half_w)
            end = min(n_samples, i + half_w + 1)
            smooth[i] = data[start:end].mean(axis=0)

        residual = np.abs(data - smooth)
        self.decision_scores_ = residual.max(axis=1)
        return self
