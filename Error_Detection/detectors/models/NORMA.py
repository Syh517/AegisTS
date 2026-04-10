import numpy as np

class NORMA:
    """
    多变量时序 NORMA 异常检测（在线核方法）
    """
    def __init__(self, kernel='rbf', gamma=0.1, eta=0.01, lam=1e-4):
        self.kernel = kernel
        self.gamma = gamma        # RBF gamma
        self.eta = eta            # 学习率
        self.lam = lam            # 正则化
        self.alpha = None         # 权重
        self.X_train = None
        self.decision_scores_ = None

    def _rbf_kernel(self, X1, X2):
        # X1: (n1, d), X2: (n2, d)
        diff = X1[:, None, :] - X2[None, :, :]
        K = np.exp(-self.gamma * np.sum(diff ** 2, axis=2))
        return K

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        self.X_train = []
        self.alpha = []

        scores = []

        for t in range(n_samples):
            x_t = data[t:t+1, :]  # shape (1, d)
            if len(self.X_train) == 0:
                score = 0.0
                self.X_train.append(x_t)
                self.alpha.append(self.eta)
                scores.append(score)
                continue

            X_stack = np.vstack(self.X_train)
            K = self._rbf_kernel(x_t, X_stack)  # shape (1, len(X_train))
            pred = np.dot(K, np.array(self.alpha))
            score = np.abs(pred - 0)  # 无监督，假设目标为0
            scores.append(score[0])

            # 在线更新 alpha
            alpha_new = (1 - self.eta * self.lam) * np.array(self.alpha)
            alpha_new = np.append(alpha_new, self.eta)
            self.alpha = list(alpha_new)
            self.X_train.append(x_t)

        self.decision_scores_ = np.array(scores)
        return self
import numpy as np

class NORMA:
    """
    多变量时序 NORMA 异常检测（在线核方法）
    """
    def __init__(self, kernel='rbf', gamma=0.1, eta=0.01, lam=1e-4):
        self.kernel = kernel
        self.gamma = gamma        # RBF gamma
        self.eta = eta            # 学习率
        self.lam = lam            # 正则化
        self.alpha = None         # 权重
        self.X_train = None
        self.decision_scores_ = None

    def _rbf_kernel(self, X1, X2):
        # X1: (n1, d), X2: (n2, d)
        diff = X1[:, None, :] - X2[None, :, :]
        K = np.exp(-self.gamma * np.sum(diff ** 2, axis=2))
        return K

    def fit(self, data: np.ndarray):
        n_samples, n_features = data.shape
        self.X_train = []
        self.alpha = []

        scores = []

        for t in range(n_samples):
            x_t = data[t:t+1, :]  # shape (1, d)
            if len(self.X_train) == 0:
                score = 0.0
                self.X_train.append(x_t)
                self.alpha.append(self.eta)
                scores.append(score)
                continue

            X_stack = np.vstack(self.X_train)
            K = self._rbf_kernel(x_t, X_stack)  # shape (1, len(X_train))
            pred = np.dot(K, np.array(self.alpha))
            score = np.abs(pred - 0)  # 无监督，假设目标为0
            scores.append(score[0])

            # 在线更新 alpha
            alpha_new = (1 - self.eta * self.lam) * np.array(self.alpha)
            alpha_new = np.append(alpha_new, self.eta)
            self.alpha = list(alpha_new)
            self.X_train.append(x_t)

        self.decision_scores_ = np.array(scores)
        return self
