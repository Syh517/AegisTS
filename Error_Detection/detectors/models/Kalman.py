import numpy as np

class Kalman:
    def __init__(self):
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        smooth = np.zeros_like(data)
        smooth[0] = data[0]

        # 简单 Kalman filter（假设单位矩阵状态转移）
        for t in range(1, n_samples):
            smooth[t] = 0.5 * data[t] + 0.5 * smooth[t-1]

        residual = np.abs(data - smooth)
        self.decision_scores_ = residual.max(axis=1)
        return self
