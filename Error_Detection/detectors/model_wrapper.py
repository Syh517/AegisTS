import numpy as np
import math
from Error_Detection.detectors.utils.slidingWindows import find_length_rank

# Unsupervise_AD_Pool = [ 'IForest', 'LOF', 'PCA', 'HBOS', 'KNN', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA']

Unsupervise_AD_Pool = ['Sub_IForest', 'IForest', 'Sub_LOF', 'LOF', 'Sub_PCA', 'PCA', 'HBOS', 'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'FFT', 'NORMA', 
                       'MovingAverage', 'EWMA', 'SavGol', 'Kalman', 'LOESS', 'HoltWinters', 'RollingMedian',
                       'ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM',
                       'SARAD', 'SensitiveHUE', 'SensorSCAN', 'CATCH', 'DADA',]

# METHODS = ['MovingAverage', 'EWMA', 'SavGol', 'Kalman', 'LOESS', 'HoltWinters', 'RollingMedian',
#             'ZScore', 'MAD', 'IQR', 'Mahalanobis', 'HotellingT2', 'PCAStat', 'KDE', 'GMM']

Methods = ['ZScore', 'MAD', 'IQR', 'MovingAverage', 'EWMA', 'RollingMedian', 'SavGol', 'HoltWinters', 'Kalman', 'LOESS']   


def run_Unsupervise_AD(model_name, data, **kwargs):
    function_name = f'run_{model_name}'
    function_to_call = globals()[function_name]
    results = function_to_call(data, **kwargs)
    return results
    # try:
    #     function_name = f'run_{model_name}'
    #     function_to_call = globals()[function_name]
    #     results = function_to_call(data, **kwargs)
    #     return results
    # except KeyError:
    #     error_message = f"Model function '{function_name}' is not defined.", 
    #     print(error_message)
    #     return error_message
    # except Exception as e:
    #     error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
    #     print(error_message)
    #     return error_message




def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from Error_Detection.detectors.models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IForest(data, slidingWindow=20, n_estimators=100, max_features=1, n_jobs=1):
    from Error_Detection.detectors.models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from Error_Detection.detectors.models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from Error_Detection.detectors.models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KShapeAD(data, periodicity=1):
    from Error_Detection.detectors.models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4*(slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5*slidingWindow))
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from Error_Detection.detectors.models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCA(data, slidingWindow=20, n_components=None, n_jobs=1):
    from Error_Detection.detectors.models.PCA import PCA
    clf = PCA(slidingWindow = slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from Error_Detection.detectors.models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from Error_Detection.detectors.models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from Error_Detection.detectors.models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from Error_Detection.detectors.models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from Error_Detection.detectors.models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_KMeansAD_U(data, n_clusters=20, periodicity=1,n_jobs=1):
    from Error_Detection.detectors.models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()

def run_COPOD(data, n_jobs=1):
    from Error_Detection.detectors.models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from Error_Detection.detectors.models.CBLOF import CBLOF
    try:
        clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
        clf.fit(data)
        score = clf.decision_scores_
    except ValueError as e:
        if "Could not form valid cluster separation" in str(e):
            # Reduce the number of clusters to ensure valid separation can be formed
            n_clusters_fallback = max(2, n_clusters // 2)  # Try with half the clusters
            try:
                clf = CBLOF(n_clusters=n_clusters_fallback, alpha=alpha, n_jobs=n_jobs)
                clf.fit(data)
                score = clf.decision_scores_
            except ValueError:
                # If still failing, return a default score array
                score = np.ones(data.shape[0]) * 0.5  # Neutral outlier score
        else:
            # Re-raise if it's a different ValueError
            raise
    return score.ravel()

def run_COF(data, n_neighbors=30):
    from Error_Detection.detectors.models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EIF(data, n_trees=100):
    from Error_Detection.detectors.models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RobustPCA(data, max_iter=1000):
    from Error_Detection.detectors.models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_FFT(data, keep_ratio=0.1):
    from Error_Detection.detectors.models.FFT import FFT
    clf = FFT(keep_ratio=keep_ratio)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_NORMA(data, kernel='rbf', gamma=0.1, eta=0.01, lam=1e-4):
    from Error_Detection.detectors.models.NORMA import NORMA
    clf = NORMA(kernel=kernel, gamma=gamma, eta=eta, lam=lam)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()





# 平滑方法

def run_MovingAverage(data, window=5):
    from Error_Detection.detectors.models.MovingAverage import MovingAverage
    clf = MovingAverage(window=window)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_EWMA(data, alpha=0.3):
    from Error_Detection.detectors.models.EWMA import EWMA
    clf = EWMA(alpha=alpha)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_SavGol(data, window=7, poly=2):
    from Error_Detection.detectors.models.SavGol import SavGol
    clf = SavGol(window=window, poly=poly)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Kalman(data):
    from Error_Detection.detectors.models.Kalman import Kalman
    clf = Kalman()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_LOESS(data, frac=0.3):
    from Error_Detection.detectors.models.LOESS import LOESS
    clf = LOESS(frac=frac)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HoltWinters(data, alpha=0.3, beta=0.1):
    from Error_Detection.detectors.models.HoltWinters import HoltWinters
    clf = HoltWinters( alpha=alpha, beta=beta)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_RollingMedian(data, window=5):
    from Error_Detection.detectors.models.RollingMedian import RollingMedian
    clf = RollingMedian(window=window)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()




# 统计方法

def run_ZScore(data):
    from Error_Detection.detectors.models.ZScore import ZScore
    clf = ZScore()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_MAD(data):
    from Error_Detection.detectors.models.MAD import MAD
    clf = MAD()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_IQR(data, k=1.5):
    from Error_Detection.detectors.models.IQR import IQR
    clf = IQR(k=k)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_Mahalanobis(data):
    from Error_Detection.detectors.models.Mahalanobis import Mahalanobis
    clf = Mahalanobis()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_HotellingT2(data):
    from Error_Detection.detectors.models.HotellingT2 import HotellingT2
    clf = HotellingT2()
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_PCAStat(data, n_components=2):
    from Error_Detection.detectors.models.PCAStat import PCAStat
    clf = PCAStat(n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_KDE(data, bandwidth=1.0):
    from Error_Detection.detectors.models.KDE import KDE
    clf = KDE(bandwidth=bandwidth)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()

def run_GMM(data, n_components=2):
    from Error_Detection.detectors.models.GMM import GMM
    clf = GMM(n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()






def run_SARAD(data, window_size=32, d_model=64):
    from Error_Detection.detectors.models.SARAD import SARAD_Model
    
    data = np.expand_dims(data, axis=1)

    clf = SARAD_Model(window_size=window_size, d_model=d_model)
    clf.fit(data)
    score = clf.decision_scores_(data)
    return score.ravel()


def run_SensitiveHUE(data, window_size=32):
    from Error_Detection.detectors.models.SensitiveHUE import SensitiveHUE_Model
    
    data = np.expand_dims(data, axis=1)

    clf = SensitiveHUE_Model(window_size=window_size)
    clf.fit(data)
    score = clf.decision_scores_(data)
    return score.ravel()


def run_SensorSCAN(data, window_size=32):
    from Error_Detection.detectors.models.SensorSCAN import SensorSCAN_Model
    
    data = np.expand_dims(data, axis=1)

    clf = SensorSCAN_Model(window_size=window_size)
    clf.fit(data)
    score = clf.decision_scores_(data)
    return score.ravel()


def run_CATCH(data, window_size=32, patch_len=8):
    from Error_Detection.detectors.models.CATCH import CATCH_Model
    
    data = np.expand_dims(data, axis=1)

    clf = CATCH_Model(window_size=window_size, patch_len=patch_len)
    clf.fit(data)
    score = clf.decision_scores_(data)
    return score.ravel()


def run_DADA(data, window_size=32):
    from Error_Detection.detectors.models.DADA import DADA_Model
    
    data = np.expand_dims(data, axis=1)

    clf = DADA_Model(window_size=window_size)
    clf.fit(data)
    score = clf.decision_scores_(data)
    return score.ravel()