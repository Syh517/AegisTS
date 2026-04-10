import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Error_Detection.detectors.model_wrapper import *
# from Error_Detection.metaod.models.predict_metaod import select_model
from Error_Detection.detectors.HP_list import algo_HP_dict
from Error_Detection.FMMS.predict_FMMS import select_model


class OutlierDetector:
    def __init__(self, n_selection=5):
        self.n_selection = n_selection

    def _sanitize_scores(self, output):
        scores = np.asarray(output, dtype=np.float64).reshape(-1, 1)

        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return np.zeros(scores.shape[0], dtype=np.float64)

        finite_values = scores[finite_mask]
        replacement = np.median(finite_values)
        scores = np.where(finite_mask, scores, replacement)

        return MinMaxScaler((0, 1)).fit_transform(scores).ravel()

    def get_admodels(self, data):
        models = select_model(data, n_selection=self.n_selection)
        return models

    def get_adscores(self, models, data):
        # data here is np.array
        ad_scores = {}
        for model_name in models:
            if model_name not in algo_HP_dict:
                print(f"[Skip] No HP defined for model: {model_name}")
                continue


            hp = algo_HP_dict[model_name]     

            output = run_Unsupervise_AD(model_name, data, **hp)

            output = self._sanitize_scores(output)

            ad_scores[model_name] = output      

        if len(ad_scores) == 0:
            return np.zeros(data.shape[0], dtype=np.float64)

        scores = np.stack(list(ad_scores.values()), axis=0)
        mean_scores = np.mean(scores, axis=0)

        return mean_scores
