import numpy as np
from scipy.signal import savgol_filter

class SavGol:
    def __init__(self, window=7, poly=2):
        self.window = window
        self.poly = poly
        self.decision_scores_ = None

    def fit(self, data: np.ndarray):
        smooth = savgol_filter(data, window_length=self.window, polyorder=self.poly, axis=0)
        residual = np.abs(data - smooth)
        self.decision_scores_ = residual.max(axis=1)
        return self
