"""
HoloClean-TS Lite (no-DB) — probabilistic + constraint-driven time series cleaner.

Features:
- Convert wide multivariate time series to internal long format and back.
- Detect violations for: Speed (S), Acceleration (A), Trend (R), Cross-variable (C), Logical (L).
- Generate candidate repairs from temporal neighbors, historical stats, and cross-var regression.
- Score candidates using likelihood (z-score), regression residuals and constraint satisfaction penalty.
- Output repaired time series in wide format with optional confidence per repaired cell.

Dependencies:
    pip install pandas numpy scipy scikit-learn
"""

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Main class
# --------------------------
class HoloClean:
    """
    Lightweight DB-free HoloClean style cleaner for multivariate time series.

    Parameters:
    -----------
    df_ts : pandas.DataFrame
        Wide format: must contain `timestamp_col` and one or more variable columns.
    timestamp_col : str
        Name of timestamp column (will be converted to datetime if possible).
    min_neighbors : int
        How many temporal neighbors to use when creating candidate set.
    """

    def __init__(self,
                 timestamp_col: str = 'timestamp',
                 min_neighbors: int = 2,
                 verbose: bool = True):
        self.timestamp_col = timestamp_col
        self.min_neighbors = min_neighbors
        self.verbose = verbose

        # placeholders to be filled in fit
        self.df_orig = None
        self.df_long = None
        self.ts_numeric = None
        self.vars = None
        # constraints structures
        self.speed_limits = {}   # var -> max speed (abs delta / delta_t)
        self.accel_limits = {}   # var -> max second-difference magnitude
        self.trend_rules = {}    # var -> expected sign: 'increasing'|'decreasing'|'none' or function
        self.cross_var_models = {}  # target_var -> LinearRegression trained on other vars
        self.logic_rules = {}    # var -> lambda v: bool
        # stats
        self.hist_mean = {}
        self.hist_std = {}

    # ----------------------
    # Data conversion
    # ----------------------
    def fit(self, df_ts: pd.DataFrame):
        """
        Fit internals (convert timestamps, compute stats).
        Does NOT modify data. Call before detect/repair.
        """
        df = df_ts.copy()
        if self.timestamp_col not in df.columns:
            raise ValueError(f"{self.timestamp_col} not in dataframe columns")

        # ensure timestamp is datetime
        try:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        except Exception:
            pass

        self.df_orig = df.reset_index(drop=True)
        self.vars = [c for c in self.df_orig.columns if c != self.timestamp_col]
        self.ts_numeric = to_numeric_timestamp(self.df_orig[self.timestamp_col])
        # long format
        rows = []
        for idx, row in self.df_orig.iterrows():
            t = row[self.timestamp_col]
            for v in self.vars:
                rows.append({'index': idx, 'timestamp': t, 'variable': v, 'value': row[v]})
        self.df_long = pd.DataFrame(rows)

        # hist stats per variable (ignore NaNs)
        for v in self.vars:
            col = self.df_orig[v].dropna().astype(float)
            if len(col) > 0:
                self.hist_mean[v] = float(col.mean())
                self.hist_std[v] = float(col.std(ddof=0)) if col.std(ddof=0) > 0 else 1.0
            else:
                self.hist_mean[v] = 0.0
                self.hist_std[v] = 1.0

        if self.verbose:
            print(f"[fit] {len(self.vars)} variables, {len(self.df_orig)} timestamps.")

    # ----------------------
    # Constraint setters
    # ----------------------
    def set_speed_limits(self, speed_limits: Dict[str, Union[float, Tuple[float, float]]]):
        """ 
        speed_limits: var -> max |delta|/delta_t (units value per second) 
        OR var -> (min_delta/delta_t, max_delta/delta_t) for range constraints
        """
        self.speed_limits = speed_limits.copy()

    def set_accel_limits(self, accel_limits: Dict[str, Union[float, Tuple[float, float]]]):
        """ 
        accel_limits: var -> max absolute second-difference (value units)
        OR var -> (min_second_diff, max_second_diff) for range constraints
        """
        self.accel_limits = accel_limits.copy()

    def set_trend_rules(self, trend_rules: Dict[str, str]):
        """ trend_rules: var -> 'increasing'/'decreasing'/'none' """
        self.trend_rules = trend_rules.copy()

    def set_logic_rules(self, logic_rules: Dict[str, callable]):
        """ logic_rules: var -> function(value) -> bool meaning is_valid """
        self.logic_rules = logic_rules.copy()

    def build_cross_var_models(self, df_ts: pd.DataFrame, target_vars: Optional[List[str]] = None):
        """
        Fit simple linear regression models for each target variable
        using other variables (synchronous cross-variable prediction).
        """
        if target_vars is None:
            target_vars = self.vars
        # drop rows with NaN
        for target in target_vars:
            X = df_ts[self.vars].copy()
            y = X.pop(target)
            mask = (~y.isna()) & (~X.isna().any(axis=1))
            if mask.sum() < 3:
                # not enough data
                continue
            model = LinearRegression()
            model.fit(X[mask], y[mask])
            # compute residual std
            preds = model.predict(X[mask])
            resid = y[mask].astype(float) - preds
            resid_std = float(np.std(resid, ddof=0)) if len(resid) > 0 else 1.0
            self.cross_var_models[target] = {'model': model, 'resid_std': max(resid_std, 1e-6)}
        if self.verbose:
            print(f"[build_cross_var_models] built {len(self.cross_var_models)} cross-var linear models.")

    # ----------------------
    # Constraint checkers
    # ----------------------
    def _check_speed_violation(self, idx: int, var: str, candidate: Optional[float] = None) -> bool:
        # check using neighbor previous or next (choose previous if exists)
        if var not in self.speed_limits:
            return False
            
        limit = self.speed_limits[var]
        t_arr = self.ts_numeric
        
        # Handle both formats: single float or tuple of (min, max)
        if isinstance(limit, (tuple, list)) and len(limit) == 2:
            min_limit, max_limit = limit
        else:
            # Convert single limit to symmetric range
            min_limit, max_limit = -abs(limit), abs(limit)
            
        if idx > 0:
            t0 = t_arr[idx-1]; v0 = self.df_orig.loc[idx-1, var]
            if pd.isna(v0):
                # fallback to next
                pass
            else:
                v1 = candidate if candidate is not None else self.df_orig.loc[idx, var]
                dt = float(t_arr[idx] - t_arr[idx-1])
                if dt <= 0:
                    return False
                speed = (float(v1) - float(v0)) / dt  # Note: not abs here anymore
                return not (min_limit <= speed <= max_limit)
                
        if idx < len(t_arr)-1:
            t2 = t_arr[idx+1]; v2 = self.df_orig.loc[idx+1, var]
            if pd.isna(v2):
                return False
            v1 = candidate if candidate is not None else self.df_orig.loc[idx, var]
            dt = float(t_arr[idx+1] - t_arr[idx])
            if dt <= 0:
                return False
            speed = (float(v2) - float(v1)) / dt  # Note: not abs here anymore
            return not (min_limit <= speed <= max_limit)
        return False

    def _check_accel_violation(self, idx: int, var: str, candidate: Optional[float] = None) -> bool:
        if var not in self.accel_limits:
            return False
            
        limit = self.accel_limits[var]
        
        # Handle both formats: single float or tuple of (min, max)
        if isinstance(limit, (tuple, list)) and len(limit) == 2:
            min_limit, max_limit = limit
        else:
            # Convert single limit to symmetric range
            min_limit, max_limit = -abs(limit), abs(limit)
            
        # need t_{i-1}, t_i, t_{i+1}
        if idx > 0 and idx < len(self.df_orig)-1:
            v_prev = self.df_orig.loc[idx-1, var]
            v_next = self.df_orig.loc[idx+1, var]
            if pd.isna(v_prev) or pd.isna(v_next):
                return False
            v_i = candidate if candidate is not None else self.df_orig.loc[idx, var]
            # second difference
            accel = float(v_next) - 2*float(v_i) + float(v_prev)
            return not (min_limit <= accel <= max_limit)
        return False

    def _check_trend_violation(self, idx: int, var: str, candidate: Optional[float] = None) -> bool:
        if var not in self.trend_rules:
            return False
        rule = self.trend_rules[var]
        if rule not in ('increasing', 'decreasing'):
            return False
        # check with previous point
        if idx == 0:
            return False
        v_prev = self.df_orig.loc[idx-1, var]
        if pd.isna(v_prev):
            return False
        v_i = candidate if candidate is not None else self.df_orig.loc[idx, var]
        if rule == 'increasing':
            return float(v_i) < float(v_prev)
        else:
            return float(v_i) > float(v_prev)

    def _check_logic_violation(self, var: str, candidate: Optional[float] = None) -> bool:
        if var not in self.logic_rules:
            return False
        func = self.logic_rules[var]
        val = candidate
        if val is None:
            # use existing values
            return False
        try:
            return not bool(func(val))
        except Exception:
            return True

    def _check_cross_var_violation(self, idx: int, var: str, candidate: Optional[float] = None) -> bool:
        if var not in self.cross_var_models:
            return False
        model_info = self.cross_var_models[var]
        model = model_info['model']
        resid_std = model_info['resid_std']
        # prepare synchronous features at same timestamp
        row = self.df_orig.loc[idx, self.vars].copy().astype(float)
        # replace this var by candidate if provided
        if candidate is not None:
            row[var] = float(candidate)
        # if any other var missing, skip
        if row.isna().any():
            return False
        X = row.drop(var).values.reshape(1, -1)  # order must match training
        # need to reconstruct feature order: sklearn trained on df[other_vars] in same column order
        # We'll assume training used self.vars order dropping target -> so columns align
        try:
            pred = model.predict(row.drop(var).values.reshape(1, -1))[0]
        except Exception:
            return False
        actual = float(row[var])
        z = abs(actual - pred) / max(resid_std, 1e-6)
        # treat violation if z too large (>3)
        return z > 3.0

    # ----------------------
    # Detection
    # ----------------------
    def detect_violations(self) -> pd.DataFrame:
        """
        Return DataFrame of suspected violations with columns:
            index, variable, current_value, reasons(list)
        """
        suspects = []
        for idx in range(len(self.df_orig)):
            for var in self.vars:
                val = self.df_orig.loc[idx, var]
                if pd.isna(val):
                    # treat missing as violation
                    reasons = ['missing']
                    suspects.append({'index': idx, 'variable': var, 'current_value': np.nan, 'reasons': reasons})
                    continue
                reasons = []
                if self._check_speed_violation(idx, var):
                    reasons.append('speed')
                if self._check_accel_violation(idx, var):
                    reasons.append('accel')
                if self._check_trend_violation(idx, var):
                    reasons.append('trend')
                if self._check_cross_var_violation(idx, var):
                    reasons.append('cross_var')
                if self._check_logic_violation(var, val):
                    reasons.append('logic')
                if reasons:
                    suspects.append({'index': idx, 'variable': var, 'current_value': float(val), 'reasons': reasons})
        df_sus = pd.DataFrame(suspects)
        if self.verbose:
            print(f"[detect] found {len(df_sus)} suspect cells.")
        return df_sus

    # ----------------------
    # Candidate generation
    # ----------------------
    def _temporal_neighbors(self, idx: int, var: str, k: int = 2) -> List[float]:
        vals = []
        n = len(self.df_orig)
        # collect up to k previous and k next non-nan values
        i = idx-1
        while i >= 0 and len(vals) < k:
            v = self.df_orig.loc[i, var]
            if not pd.isna(v):
                vals.append(float(v))
            i -= 1
        i = idx+1
        while i < n and len(vals) < 2*k:
            v = self.df_orig.loc[i, var]
            if not pd.isna(v):
                vals.append(float(v))
            i += 1
        return vals

    def _historical_candidates(self, var: str, top_k: int = 5) -> List[float]:
        # propose candidates around mean ± std multipliers
        mu = self.hist_mean[var]
        sigma = max(self.hist_std[var], 1e-6)
        cand = [mu, mu+sigma, mu-sigma, mu+2*sigma, mu-2*sigma]
        return cand[:top_k]

    def _cross_var_prediction(self, idx: int, var: str) -> Optional[float]:
        if var not in self.cross_var_models:
            return None
        model_info = self.cross_var_models[var]
        model = model_info['model']
        row = self.df_orig.loc[idx, self.vars].copy().astype(float)
        if row.drop(var).isna().any():
            return None
        X = row.drop(var).values.reshape(1, -1)
        try:
            return float(model.predict(X)[0])
        except Exception:
            return None

    def generate_candidates(self, suspect_row: Dict) -> List[float]:
        """
        suspect_row: {'index': idx, 'variable': var, 'current_value': val, 'reasons': [...]}
        Returns candidate numeric values (unique, sorted by heuristic)
        """
        idx = suspect_row['index']; var = suspect_row['variable']
        cands = []
        # temporal neighbors
        neigh = self._temporal_neighbors(idx, var, k=self.min_neighbors)
        cands.extend(neigh)
        # historical
        cands.extend(self._historical_candidates(var, top_k=5))
        # cross-var pred
        pred = self._cross_var_prediction(idx, var)
        if pred is not None:
            cands.append(pred)
        # include current if not nan
        cur = suspect_row.get('current_value', None)
        if cur is not None and not pd.isna(cur):
            cands.append(float(cur))
        # uniqueness & filter NaN
        cands = [float(x) for x in cands if x is not None and (not pd.isna(x))]
        # sort by proximity to historical mean
        cands = list(dict.fromkeys(cands))  # preserve order but unique
        cands.sort(key=lambda x: abs(x - self.hist_mean.get(var, 0.0)))
        return cands

    # ----------------------
    # Scoring & repair
    # ----------------------
    def _score_candidate(self, idx: int, var: str, candidate: float, reasons: List[str]) -> Tuple[float, Dict]:
        """
        Return (score, details). Higher score = better.
        Components:
         - hist_likelihood: based on z-score to historical mean (Gaussian)
         - cross_var_likelihood: based on regression residual z
         - constraint_penalty: negative penalty if candidate violates constraints
        """
        # hist likelihood (log-likelihood)
        mu = self.hist_mean.get(var, 0.0)
        sigma = max(self.hist_std.get(var, 1.0), 1e-6)
        z_hist = (candidate - mu) / sigma
        logp_hist = norm.logpdf(z_hist)  # note: std-normal at z_hist
        # cross-var likelihood (if model exists)
        logp_cross = 0.0
        cross_penalty = 0.0
        if var in self.cross_var_models:
            info = self.cross_var_models[var]
            model = info['model']; resid_std = info['resid_std']
            row = self.df_orig.loc[idx, self.vars].copy().astype(float)
            if not row.drop(var).isna().any():
                # set candidate and predict
                row[var] = candidate
                try:
                    pred = model.predict(row.drop(var).values.reshape(1, -1))[0]
                    z_cross = (candidate - pred) / max(resid_std, 1e-6)
                    logp_cross = norm.logpdf(z_cross)
                except Exception:
                    logp_cross = -5.0
            else:
                logp_cross = -1.0

        # constraints checks: each violated constraint imposes penalty
        penalty = 0.0
        violated = []
        # speed
        if self._check_speed_violation(idx, var, candidate):
            penalty -= 8.0
            violated.append('speed')
        if self._check_accel_violation(idx, var, candidate):
            penalty -= 6.0
            violated.append('accel')
        if self._check_trend_violation(idx, var, candidate):
            penalty -= 4.0
            violated.append('trend')
        if self._check_cross_var_violation(idx, var, candidate):
            penalty -= 7.0
            violated.append('cross_var')
        if self._check_logic_violation(var, candidate):
            penalty -= 10.0
            violated.append('logic')

        # aggregate score
        score = (logp_hist * 1.0) + (logp_cross * 0.8) + penalty
        details = {
            'logp_hist': float(logp_hist),
            'logp_cross': float(logp_cross),
            'penalty': float(penalty),
            'violated': violated
        }
        return float(score), details

    def repair(self, max_candidates: int = 10, min_score_gap: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full repair pipeline:
            1) detect violations
            2) for each suspect, generate candidates
            3) score candidates, pick best
        Returns:
            df_repaired_wide, df_changes  (changes: index, var, old, new, score, details)
        """
        suspects = self.detect_violations()
        changes = []
        df_repaired = self.df_orig.copy()

        for _, s in suspects.iterrows():
            sdict = s.to_dict()
            idx = int(sdict['index']); var = sdict['variable']; reasons = sdict['reasons']
            cands = self.generate_candidates(sdict)
            if len(cands) == 0:
                continue
            # limit candidates
            cands = cands[:max_candidates]
            scored = []
            for c in cands:
                sc, det = self._score_candidate(idx, var, c, reasons)
                scored.append((sc, c, det))
            # also consider leaving as NaN (if originally NaN)
            if pd.isna(sdict['current_value']):
                # propose mean
                sc_mean, det_mean = self._score_candidate(idx, var, self.hist_mean[var], reasons)
                scored.append((sc_mean, self.hist_mean[var], det_mean))

            # pick best score
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_val, best_det = scored[0]
            # optional: ensure best is meaningfully better than current (if current exists)
            cur_val = sdict['current_value']
            cur_score = None
            if cur_val is not None and (not pd.isna(cur_val)):
                cur_score, _ = self._score_candidate(idx, var, float(cur_val), reasons)
            if cur_score is not None:
                if best_score - cur_score < min_score_gap:
                    # not confident to change
                    if self.verbose:
                        # skip change
                        pass
                    continue
            # apply change if improved
            if cur_score is None or best_score > cur_score + 1e-6:
                old = df_repaired.loc[idx, var]
                df_repaired.loc[idx, var] = best_val
                changes.append({
                    'index': idx,
                    'timestamp': df_repaired.loc[idx, self.timestamp_col],
                    'variable': var,
                    'old': old,
                    'new': best_val,
                    'score': best_score,
                    'details': best_det
                })
                if self.verbose:
                    print(f"[repair] idx={idx} var={var} {old} -> {best_val} (score={best_score:.2f})")

        df_changes = pd.DataFrame(changes)
        # return repaired wide dataframe and change log
        return df_repaired.reset_index(drop=True), df_changes

# --------------------------
# Helper functions
# --------------------------
def to_numeric_timestamp(series: pd.Series) -> np.ndarray:
    # convert to float seconds since epoch for difference computations
    if np.issubdtype(series.dtype, np.datetime64):
        return series.astype('int64') / 1e9
    else:
        try:
            s = pd.to_datetime(series)
            return s.astype('int64') / 1e9
        except Exception:
            # assume numeric already
            return series.astype(float)

# --------------------------
# Example usage
# --------------------------
if __name__ == '__main__':
    # simple demo with synthetic data
    import io
    csv = """timestamp,temp,pressure,humidity
2025-10-01 00:00:00,20.0,101.0,40
2025-10-01 00:01:00,20.5,101.1,41
2025-10-01 00:02:00,1000.0,101.2,42
2025-10-01 00:03:00,21.0,2000.0,43
2025-10-01 00:04:00,21.5,101.4,44
"""
    df = pd.read_csv(io.StringIO(csv), parse_dates=['timestamp'])
    cleaner = HoloClean(timestamp_col='timestamp', min_neighbors=1, verbose=True)
    cleaner.fit(df)

    # set constraints (example units: value per second)
    # here timestamps spaced by 60s -> speed limit e.g., 0.05 units per second => 3 units per minute
    # Using new format with tuple (min, max)
    cleaner.set_speed_limits({
        'temp': (-2.0, 2.0),
        'pressure': (-1.0, 1.0),
        'humidity': (-3.0, 3.0)
    })
    cleaner.set_accel_limits({
        'temp': (-5.0, 5.0),
        'pressure': (-10.0, 10.0),
        'humidity': (-7.0, 7.0)
    })
    cleaner.set_trend_rules({'temp': 'increasing'})
    cleaner.set_logic_rules({'humidity': lambda x: 0 <= x <= 100})
    cleaner.build_cross_var_models(df)

    repaired_df, changes = cleaner.repair()
    print("\nOriginal:")
    print(df)
    print("\nRepaired:")
    print(repaired_df)
    print("\nChanges:")
    print(changes)
