import numpy as np

class FFT:
    """
    多变量时序异常检测（FFT残差法）
    - data: numpy.ndarray, shape (n_samples, n_features)
    - keep_ratio: 保留低频比例
    """
    def __init__(self, keep_ratio=0.1):
        self.keep_ratio = keep_ratio
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        recon = np.zeros_like(data)

        for i in range(n_features):
            X = np.fft.fft(data[:, i])
            n_keep = max(1, int(self.keep_ratio * len(X)))
            # 保留低频，其余置零
            X[n_keep:-n_keep] = 0
            recon[:, i] = np.fft.ifft(X).real

        # 计算每行 max 残差作为异常分数
        residual = np.abs(data - recon)
        self.decision_scores_ = np.max(residual, axis=1)
        return self
