import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesHoloClean:
    """
    A practical "HoloClean-like" cleaner for multivariate time series.
    Key ideas:
      - For each corrupted cell (label==1), generate a set of candidate repairs:
        * temporal interpolation / smoothing
        * multivariate regression prediction (using other vars & neighbors)
        * global statistic (median/mean)
      - Score candidates using factors:
        * temporal smoothness (adjacent timestamps)
        * agreement with multivariate regression residuals
        * domain constraints (optional)
      - Use iterative EM-like weight estimation to combine scores and pick MAP.
    Usage:
      cleaner = TimeSeriesHoloClean(...)
      data_repaired = cleaner.fit_transform(data_abnormal, label)
    """

    def __init__(
        self,
        max_iter: int = 5,
        alpha_reg: float = 1.0,
        temporal_window: int = 3,
        candidate_local_window: int = 5,
        verbose: bool = True,
        constraint_funcs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param max_iter: maximum EM-style iterations
        :param alpha_reg: ridge regularization for regression model
        :param temporal_window: for temporal smoothness factor (how many neighbors)
        :param candidate_local_window: when generating interpolation/regression candidates
        :param verbose: show progress
        :param constraint_funcs: optional dict mapping column->callable(value)->bool for hard constraints
        """
        self.max_iter = max_iter
        self.alpha_reg = alpha_reg
        self.temporal_window = temporal_window
        self.candidate_local_window = candidate_local_window
        self.verbose = verbose
        self.constraint_funcs = constraint_funcs or dict()

        # internals
        self.models = dict()  # per-column regression models
        self.scaler = None
        self.global_stats = dict()

    # -------------------------
    # Utility / candidate funcs
    # -------------------------
    @staticmethod
    def _linear_interp_col(series: pd.Series, idx: int, window: int):
        """Simple linear interpolation using nearest valid neighbors within window."""
        n = len(series)
        left = None
        right = None
        # find left valid
        for d in range(1, window + 1):
            i = idx - d
            if i < 0:
                break
            if not pd.isna(series.iat[i]):
                left = (i, float(series.iat[i]))
                break
        # find right valid
        for d in range(1, window + 1):
            i = idx + d
            if i >= n:
                break
            if not pd.isna(series.iat[i]):
                right = (i, float(series.iat[i]))
                break
        if left is None and right is None:
            return None
        if left is None:
            return right[1]
        if right is None:
            return left[1]
        # linear interpolation
        x0, y0 = left
        x1, y1 = right
        if x1 == x0:
            return (y0 + y1) / 2.0
        t = (idx - x0) / (x1 - x0)
        return y0 * (1 - t) + y1 * t

    @staticmethod
    def _rolling_median_col(series: pd.Series, idx: int, halfwin: int):
        n = len(series)
        L = max(0, idx - halfwin)
        R = min(n, idx + halfwin + 1)
        window = series.iloc[L:R].dropna()
        if window.empty:
            return None
        return float(window.median())

    def _fit_regressors(self, X_df: pd.DataFrame, mask_good: pd.DataFrame):
        """
        Fit per-column regression models that predict each variable from other variables (same timestamp)
        + small temporal features (lag/lead averages).
        Use only rows where all involved columns are good (not masked).
        """
        cols = X_df.columns.tolist()
        n = len(X_df)

        # prepare lag features (mean of neighboring window)
        lag_feats = {}
        w = self.candidate_local_window
        for c in cols:
            arr = X_df[c].to_numpy(dtype=float)
            pad = np.full(n, np.nan)
            for i in range(n):
                L = max(0, i - w)
                R = min(n, i + w + 1)
                window = arr[L:R]
                pad[i] = np.nanmean(window) if np.isfinite(window).any() else np.nan
            lag_feats[c] = pad

        # Build dataset per target column
        self.models = {}
        for target in cols:
            # rows where target is good and most predictors are good
            rows = []
            Xs = []
            Ys = []
            for i in range(n):
                if pd.isna(X_df[target].iat[i]):
                    continue
                # require target good at this row and at least half predictors good
                good_preds = 0
                feat = []
                for other in cols:
                    if other == target:
                        continue
                    v = X_df[other].iat[i]
                    if pd.isna(v):
                        feat.append(np.nan)
                    else:
                        feat.append(float(v))
                        good_preds += 1
                # add lag of target
                feat.append(lag_feats[target][i])
                if good_preds >= max(1, (len(cols) - 1) // 2) and not np.isnan(feat[-1]):
                    Xs.append(feat)
                    Ys.append(float(X_df[target].iat[i]))
            if len(Ys) < 10:
                # not enough data to fit; leave model None
                self.models[target] = None
                continue
            Xs = np.array(Xs, dtype=float)
            Ys = np.array(Ys, dtype=float)
            imp = SimpleImputer(strategy="mean")
            Xs = imp.fit_transform(Xs)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xs)
            model = Ridge(alpha=self.alpha_reg)
            model.fit(Xs, Ys)
            # save pipeline pieces
            self.models[target] = {"model": model, "imputer": imp, "scaler": scaler, "cols": cols}
        if self.verbose:
            fitted = sum(1 for v in self.models.values() if v is not None)
            print(f"[fit_regressors] fitted {fitted}/{len(cols)} column regressors")

    def _predict_regressor(self, X_df: pd.DataFrame, idx: int, target: str):
        """Predict target at row idx using its regressor if available."""
        meta = self.models.get(target)
        if meta is None:
            return None
        cols = meta["cols"]
        feat = []
        for other in cols:
            if other == target:
                continue
            v = X_df[other].iat[idx]
            feat.append(np.nan if pd.isna(v) else float(v))
        # add lag mean for target
        w = self.candidate_local_window
        n = len(X_df)
        L = max(0, idx - w)
        R = min(n, idx + w + 1)
        window = X_df[target].iloc[L:R].dropna().to_numpy(dtype=float)
        if window.size == 0:
            lag = np.nan
        else:
            lag = float(window.mean())
        feat.append(lag)
        Xv = np.array(feat).reshape(1, -1)
        Xv = meta["imputer"].transform(Xv)
        Xv = meta["scaler"].transform(Xv)
        pred = float(meta["model"].predict(Xv)[0])
        return pred

    # -------------------------
    # Scoring / factor functions
    # -------------------------
    def _score_temporal(self, series: pd.Series, idx: int, candidate: float):
        """Score temporal smoothness: small difference to neighbors => higher score."""
        n = len(series)
        winsz = self.temporal_window
        neigh = []
        for d in range(1, winsz + 1):
            if idx - d >= 0 and not pd.isna(series.iat[idx - d]):
                neigh.append(float(series.iat[idx - d]))
            if idx + d < n and not pd.isna(series.iat[idx + d]):
                neigh.append(float(series.iat[idx + d]))
        if len(neigh) == 0:
            return 0.0
        diff = np.abs(np.array(neigh) - candidate)
        # Gaussian-shaped score
        sigma = np.std(neigh) if np.std(neigh) > 1e-6 else 1.0
        score = - np.mean(diff) / (sigma + 1e-6)
        return score

    def _score_regression(self, X_df: pd.DataFrame, idx: int, col: str, candidate: float):
        """
        Score agreement with regressor prediction: negative absolute residual (higher better).
        We temporarily substitute candidate into X_df for target to compute prediction residual across variables.
        """
        # If we have regressor for this column, compare
        pred = self._predict_regressor(X_df, idx, col)
        if pred is None:
            return 0.0
        res = np.abs(pred - candidate)
        sigma = max(1e-3, np.std([pred, candidate]))
        return -res / sigma

    def _satisfy_constraints(self, col: str, value: float):
        """
        Hard constraint satisfaction. If constraint fails, return a large negative penalty.
        constraint_funcs[col] should be callable(value)->bool (True if ok)
        """
        fn = self.constraint_funcs.get(col)
        if fn is None:
            return 0.0
        ok = fn(value)
        return 0.0 if ok else -100.0

    # -------------------------
    # Main API
    # -------------------------
    def fit_transform(self, data_abnormal: pd.DataFrame, label: pd.Series):
        """
        :param data_abnormal: pd.DataFrame, index is time-ordered (0..n-1)
        :param label: pd.Series of same length; label==1 rows indicate 'some variable(s) in this row corrupted'
        :return: data_repaired (copy)
        """
        assert isinstance(data_abnormal, pd.DataFrame)
        assert len(label) == len(data_abnormal)

        data = data_abnormal.copy().astype(float)
        n, m = data.shape
        cols = data.columns.tolist()

        # mask of corrupted cells: if a row labeled 1 -> treat all columns in that row as candidate corrupted cells
        error_rows = (label == 1).to_numpy().astype(bool)
        corrupted_mask = np.zeros_like(data.values, dtype=bool)
        for i in range(n):
            if error_rows[i]:
                # Mark columns with NaN or everything? Here mark all as candidate corrupted (but if some cell is known correct and not NaN, we still consider it)
                corrupted_mask[i, :] = True

        # Global stats
        for c in cols:
            arr = data[c].dropna().to_numpy(dtype=float)
            if arr.size == 0:
                self.global_stats[c] = {"mean": np.nan, "median": np.nan}
            else:
                self.global_stats[c] = {"mean": float(np.mean(arr)), "median": float(np.median(arr))}

        # initial imputation (for building regressors)
        imputer = SimpleImputer(strategy="mean")
        data_init = pd.DataFrame(imputer.fit_transform(data), columns=cols)

        # iteratively refine:
        weights = {"temporal": 1.0, "regression": 1.0, "global": 0.5, "constraint": 10.0}
        for it in range(self.max_iter):
            if self.verbose:
                print(f"[iter {it+1}/{self.max_iter}] building regressors and proposing candidates...")

            # fit regressors on current imputed data, using only rows not labeled corrupted (or where we trust)
            trust_mask = ~error_rows
            X_trust = data_init.copy()
            # zero-out rows we don't trust to nan so fit ignores them
            X_trust.loc[~trust_mask, :] = np.nan
            self._fit_regressors(X_trust, None)

            # for each corrupted cell, generate candidates
            proposals = dict()  # key: (i,col)->list of candidates
            for i in range(n):
                if not error_rows[i]:
                    continue
                for col in cols:
                    key = (i, col)
                    proposals[key] = []
                    s = data[col]
                    # 1) linear interpolation candidate
                    v_lin = self._linear_interp_col(s, i, self.candidate_local_window)
                    if v_lin is not None:
                        proposals[key].append(("interp", v_lin))
                    # 2) rolling median candidate
                    v_med = self._rolling_median_col(s, i, self.candidate_local_window)
                    if v_med is not None:
                        proposals[key].append(("median", v_med))
                    # 3) regression candidate
                    v_reg = self._predict_regressor(data_init, i, col)
                    if v_reg is not None:
                        proposals[key].append(("reg", v_reg))
                    # 4) global stat candidate
                    v_glob = self.global_stats[col]["median"]
                    if not np.isnan(v_glob):
                        proposals[key].append(("global", v_glob))
                    # deduplicate by rounded value (to avoid duplicates)
                    seen = set()
                    uniq = []
                    for t, val in proposals[key]:
                        kround = round(float(val), 9)
                        if kround in seen:
                            continue
                        seen.add(kround)
                        uniq.append((t, float(val)))
                    proposals[key] = uniq

            # Score candidates and pick highest prob (soft weights -> compute posterior-like scores)
            updates = []
            for (i, col), cands in proposals.items():
                if len(cands) == 0:
                    continue
                scores = []
                for tag, cand in cands:
                    sc_tem = weights["temporal"] * self._score_temporal(data[col], i, cand)
                    sc_reg = weights["regression"] * self._score_regression(data_init, i, col, cand)
                    sc_glob = weights["global"] * ( - abs(cand - self.global_stats[col]["median"]) / (abs(self.global_stats[col]["median"]) + 1e-6) )
                    sc_con = weights["constraint"] * self._satisfy_constraints(col, cand)
                    total = sc_tem + sc_reg + sc_glob + sc_con
                    scores.append(total)
                # softmax to posterior-ish
                maxs = max(scores)
                exps = np.exp(np.array(scores) - maxs)
                probs = exps / (exps.sum() + 1e-12)
                # choose argmax for MAP replacement (we can also do probabilistic sampling)
                idx_choice = int(np.argmax(probs))
                chosen_val = cands[idx_choice][1]
                updates.append(((i, col), chosen_val, probs[idx_choice], cands[idx_choice][0]))

            # apply updates to data_init (but do not overwrite already non-corrupted rows)
            for (i, col), val, prob, tag in updates:
                data_init.at[i, col] = val

            # optionally adjust weights (very simple heuristic): if regression dominated, increase regression weight etc.
            # compute average reg score vs temporal score
            if len(updates) > 0:
                avg_reg = np.mean([self._score_regression(data_init, i, col, val) for ((i, col), val, *_ ) in [
                    (u[0], u[1], u[2]) for u in [(x[0], x[1], x[2]) for x in updates]]]) if len(updates)>0 else 0.0
                # We keep weight adjustment mild to maintain stability
                # (in a fuller implementation we would do proper EM / M-step)
                # small damping update:
                # weights["regression"] *= 1.0 + 0.05 * np.tanh(avg_reg)
                # keep simple (no change), but we could add heuristics here

            if self.verbose:
                print(f"[iter {it+1}] applied {len(updates)} updates")

        # Final selection: map into result DataFrame; only change corrupted cells
        data_repaired = data_abnormal.copy().astype(float)
        for i in range(n):
            if not error_rows[i]:
                continue
            for col in cols:
                val = data_init.at[i, col]
                data_repaired.at[i, col] = val

        return data_repaired


# -------------------------
# Small test / demo
# -------------------------
def _make_synthetic(n=200, seed=42):
    """Make a synthetic 3-var multivariate time series with trend+seasonality+noise"""
    np.random.seed(seed)
    t = np.arange(n)
    x1 = 0.02 * t + 2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, n)
    x2 = 0.5 * x1 + 0.5 * np.cos(2 * np.pi * t / 12) + np.random.normal(0, 0.3, n)
    x3 = 0.3 * x1 - 0.2 * x2 + np.random.normal(0, 0.4, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return df


def _inject_anomalies(df: pd.DataFrame, frac_rows=0.1, frac_cells=0.6, seed=1):
    np.random.seed(seed)
    n = len(df)
    n_bad_rows = max(1, int(n * frac_rows))
    bad_row_idx = np.random.choice(n, size=n_bad_rows, replace=False)
    label = pd.Series(0, index=df.index)
    df_noisy = df.copy()
    for r in bad_row_idx:
        label.iat[r] = 1
        for c in df.columns:
            if np.random.rand() < frac_cells:
                # inject either outlier or NaN randomly
                if np.random.rand() < 0.5:
                    df_noisy.at[r, c] = df_noisy.at[r, c] + np.random.normal(10.0, 5.0)
                else:
                    df_noisy.at[r, c] = np.nan
    return df_noisy, label


if __name__ == "__main__":
    # make data
    orig = _make_synthetic(300)
    noisy, label = _inject_anomalies(orig, frac_rows=0.12, frac_cells=0.7)
    print("Original head:\n", orig.head())
    print("Noisy head:\n", noisy.head())
    print("Label sum (rows corrupted):", int(label.sum()))

    # constraint example: x1 must be > -10 (silly example)
    cons = {"x1": lambda v: v > -10}

    cleaner = TimeSeriesHoloClean(max_iter=4, verbose=True, constraint_funcs=cons)
    repaired = cleaner.fit_transform(noisy, label)
    # print comparisons for first few corrupted rows
    bad_idx = np.where(label.values == 1)[0][:6]
    for i in bad_idx:
        print(f"\nRow {i}:")
        print("  orig:", orig.iloc[i].to_dict())
        print("  noisy:", noisy.iloc[i].to_dict())
        print("  repaired:", repaired.iloc[i].to_dict())

    # Evaluate simple RMSE on corrupted rows
    mask = label == 1
    rmse = np.sqrt(((repaired[mask] - orig[mask]) ** 2).mean().mean())
    print(f"\nRMSE on corrupted rows (averaged over variables): {rmse:.4f}")
