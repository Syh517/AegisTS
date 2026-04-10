import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

class GMM:
    """
    Gaussian Mixture Model for Unsupervised Anomaly Detection.

    The anomaly score is defined as the negative log-likelihood under the fitted GMM.
    Lower likelihood (higher -log p(x)) indicates higher anomaly score.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components (clusters). Should be <= n_samples.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Type of covariance matrix.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Helps ensure positive definiteness.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed for initialization.
    max_iter : int, default=100
        Maximum number of EM iterations.
    n_init : int, default=1
        Number of initializations to perform (best result kept).
    standardize : bool, default=True
        Whether to standardize the data before fitting.
    remove_duplicates : bool, default=False
        Whether to remove duplicate samples before fitting.

    Attributes
    ----------
    gmm_ : GaussianMixture
        Fitted sklearn GaussianMixture instance.
    decision_scores_ : np.ndarray of shape (n_samples,)
        Anomaly scores (negative log-likelihood) for training data.
    """

    def __init__(
        self,
        n_components=2,
        covariance_type="full",
        reg_covar=1e-3,
        random_state=None,
        max_iter=100,
        n_init=1,
        standardize=True,
        remove_duplicates=False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.standardize = standardize
        self.remove_duplicates = remove_duplicates

        self.gmm_ = None
        self.decision_scores_ = None
        self.scaler_ = None

    def fit(self, X):
        """
        Fit the GMM on the input data and compute anomaly scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=1)

        n_samples, n_features = X.shape

        if self.n_components > n_samples:
            raise ValueError(
                f"n_components ({self.n_components}) cannot be greater than "
                f"number of samples ({n_samples})."
            )

        # 数据标准化
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # 可选地移除重复样本
        if self.remove_duplicates:
            X_unique, unique_indices = np.unique(X, axis=0, return_index=True)
            if len(X_unique) < len(X):
                print(f"Warning: Removed {len(X) - len(X_unique)} duplicate samples.")
                X = X_unique

        # 更新样本数量
        n_samples = X.shape[0]
        
        # 如果组件数大于样本数，调整组件数
        if self.n_components > n_samples:
            print(f"Warning: Reducing n_components from {self.n_components} to {n_samples}")
            self.n_components = n_samples
            
        # 如果只有一个样本，直接返回
        if n_samples == 1:
            self.decision_scores_ = np.array([0.0])
            return self

        # 尝试不同的参数组合以提高鲁棒性
        reg_covar_values = [self.reg_covar, 1e-2, 1e-1, 1.0]
        covariance_types = [self.covariance_type, 'diag', 'spherical']
        
        best_gmm = None
        best_score = -np.inf
        best_reg_covar = self.reg_covar
        best_covariance_type = self.covariance_type
        
        for cov_type in covariance_types:
            # 对于'diag'和'spherical'协方差类型，我们不需要那么多组件
            n_comp = min(self.n_components, n_samples)
            if cov_type in ['diag', 'spherical']:
                n_comp = min(n_comp, n_features)
                
            if n_comp == 0:
                continue
                
            for reg_val in reg_covar_values:
                try:
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        covariance_type=cov_type,
                        reg_covar=reg_val,
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        n_init=self.n_init,
                    )
                    
                    gmm.fit(X)
                    score = gmm.score(X)  # 使用似然得分作为评估标准
                    
                    if score > best_score:
                        best_score = score
                        best_gmm = gmm
                        best_reg_covar = reg_val
                        best_covariance_type = cov_type
                        
                except Exception as e:
                    # 忽略单个尝试失败的情况，继续尝试其他参数
                    continue
        
        # 如果没有找到有效的参数组合，抛出异常
        if best_gmm is None:
            raise RuntimeError(
                "GMM fitting failed due to numerical issues. "
                "Try: (1) checking data quality, (2) reducing dimensionality, "
                "(3) increasing data size, or (4) preprocessing data differently."
            )
            
        self.gmm_ = best_gmm
        
        # 输出使用的最佳参数
        if (best_reg_covar != self.reg_covar or 
            best_covariance_type != self.covariance_type):
            print(f"Info: Using reg_covar={best_reg_covar}, covariance_type={best_covariance_type}")

        # Compute anomaly scores: negative log-likelihood
        log_prob = self.gmm_.score_samples(X)
        self.decision_scores_ = -log_prob

        return self

    def decision_function(self, X):
        """
        Compute anomaly scores for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores (higher = more anomalous).
        """
        if self.gmm_ is None:
            raise NotFittedError("This GMM instance is not fitted yet. Call 'fit' first.")

        X = check_array(X, ensure_min_features=1)
        
        # 应用相同的标准化变换
        if self.standardize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
            
        log_prob = self.gmm_.score_samples(X)
        return -log_prob