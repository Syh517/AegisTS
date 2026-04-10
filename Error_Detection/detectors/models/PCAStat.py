import numpy as np
from sklearn.decomposition import PCA

class PCAStat:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(data)
        X_recon = pca.inverse_transform(X_pca)
        residual = np.abs(data - X_recon)
        self.decision_scores_ = np.max(residual, axis=1)
        return self
