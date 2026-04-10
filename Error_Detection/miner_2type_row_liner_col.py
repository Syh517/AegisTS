import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import itertools
from sklearn import linear_model
import difflib
import math
from sklearn.linear_model import QuantileRegressor, Lasso, ElasticNet
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from sklearn.linear_model import LinearRegression
from scipy.stats import iqr
import warnings
warnings.filterwarnings('ignore')


class RowConstraintMiner:
    """
    极速版行约束挖掘器：支持全局挖掘(mine_3d)与独立样本挖掘(mine_3d_per_sample)。
    专为 3D 张量设计，优化了多样本遍历性能。
    """

    def __init__(
        self,
        degree: int = 1,
        max_features: int = 2,
        coverage: float = 0.95,
        max_residual_std: float = 0.2,
        min_r2: float = 0.9,
    ):
        self.degree = degree
        self.max_features = max_features
        self.coverage = coverage
        self.max_residual_std = max_residual_std
        self.min_r2 = min_r2

    def mine_3d(self, data_3d: np.ndarray) -> List[Dict[str, Any]]:
        """
        算法：全局合并挖掘。
        将所有样本纵向堆叠后挖掘，返回适用于全量数据的单一约束集合。
        """
        arr = np.asarray(data_3d)
        n_samples, n_ts, n_feats = arr.shape
        arr_combined = arr.reshape(-1, n_feats)
        
        # 排除 timestamp 列（假设第一列是时间）进行挖掘
        df_combined = pd.DataFrame(arr_combined[:, 1:], 
                                   columns=[f"col_{j}" for j in range(1, n_feats)])
        return self.row_miner(df_combined)

    def mine_3d_per_sample(self, data_3d: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        算法：独立样本挖掘。
        返回 List[List[Dict]]，外层列表长度为 n_samples。
        """
        arr = np.asarray(data_3d)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        n_samples = arr.shape[0]
        n_feats = arr.shape[2]
        col_names = [f"col_{j}" for j in range(1, n_feats)]
        
        all_sample_constraints = []

        for i in range(n_samples):
            # 提取单个样本并跳过时间列
            sample_data = arr[i, :, 1:] 
            df_sample = pd.DataFrame(sample_data, columns=col_names).dropna()
            
            if len(df_sample) < (self.max_features + 2):
                all_sample_constraints.append([])
                continue
                
            try:
                # 核心挖掘逻辑
                found = self.row_miner(df_sample)
                all_sample_constraints.append(found)
            except Exception:
                all_sample_constraints.append([])

        return all_sample_constraints

    def row_miner(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """高效核心挖掘器：基于线性代数加速"""
        cols = df.columns.tolist()
        constraints = []
        # 预计算相关性矩阵以减少无效组合搜索
        corr_matrix = df.corr().abs()

        for target in cols:
            # 筛选与 target 最相关的候选特征
            candidates = corr_matrix[target].sort_values(ascending=False).index[1:self.max_features + 2].tolist()
            
            for n in range(1, self.max_features + 1):
                for feat_combo in combinations(candidates, n):
                    X = df[list(feat_combo)].values
                    y = df[target].values
                    
                    # 线性化特征构建 (极简版实现 degree 1 or 2)
                    X_poly, feature_names = self._get_poly_features(X, list(feat_combo))
                    
                    # 使用 numpy.linalg.lstsq (SVD) 获得极速解析解
                    # 增加截距项
                    X_mat = np.column_stack([np.ones(X_poly.shape[0]), X_poly])
                    try:
                        coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
                    except np.linalg.LinAlgError:
                        continue

                    pred = X_mat @ coeffs
                    res = y - pred
                    
                    # 计算 R2 和 Std
                    ss_res = np.sum(res**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    res_std = np.std(res)

                    if r2 >= self.min_r2 and res_std <= self.max_residual_std:
                        # 确定残差范围 (quantile)
                        lower = np.quantile(res, (1 - self.coverage) / 2)
                        upper = np.quantile(res, 1 - (1 - self.coverage) / 2)
                        
                        model_terms = dict(zip(feature_names, coeffs[1:]))
                        intercept = coeffs[0]
                        
                        formula = f"{target} = {self._format_formula(model_terms, intercept)} | residual ∈ [{lower:.6f}, {upper:.6f}]"
                        
                        constraints.append({
                            "formula": formula,
                            "target": target,
                            "features": list(feat_combo),
                            "model_terms": model_terms,
                            "intercept": intercept,
                            "lower": lower,
                            "upper": upper,
                            "R2": float(r2),
                            "loss_mse": float(ss_res / len(y)),
                            "residual_std": float(res_std)
                        })
        
        # 每个样本内部按 R2 排序，保留最优关系
        constraints.sort(key=lambda x: -x["R2"])
        constraints = self._refine_constraints(constraints)
        return constraints

    def _get_poly_features(self, X, names):
        """手动构建特征以减少 PolynomialFeatures 的开销"""
        if self.degree == 1:
            return X, names
        
        # Degree 2 扩展: [x1, x2, x1^2, x2^2, x1*x2]
        res_X = [X]
        res_names = [n for n in names]
        for i in range(len(names)):
            res_X.append((X[:, i]**2).reshape(-1, 1))
            res_names.append(f"{names[i]}^2")
            for j in range(i + 1, len(names)):
                res_X.append((X[:, i] * X[:, j]).reshape(-1, 1))
                res_names.append(f"{names[i]}*{names[j]}")
        return np.hstack(res_X), res_names

    def _format_formula(self, terms, intercept) -> str:
        parts = []
        for name, coef in sorted(terms.items()):
            if abs(coef) < 1e-7: continue
            parts.append(f"{'+' if coef >= 0 else '-'} {abs(coef):.4f}*{name}")
        expr = " ".join(parts).strip()
        if expr.startswith("+"): expr = expr[2:]
        return f"{intercept:.4f} + {expr}" if abs(intercept) > 1e-7 else expr or "0.0000"
    
    def _refine_constraints(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        精简约束逻辑：
        1. 每个 target 只保留 Loss 最小的一个。
        2. 剔除系数过小的权重项。
        """
        if not constraints: return []
        
        df_c = pd.DataFrame(constraints)
        # 按 target 分组，保留 loss_mse 最小的行
        best_indices = df_c.groupby("target")["loss_mse"].idxmin()
        refined = df_c.loc[best_indices].to_dict("records")
        
        # 二次精简：如果公式右侧包含多个互为镜像的变量，进一步手动筛选（可选）
        return refined


class ColConstraintMiner:
    """三维列约束挖掘器：处理(n_samples, n_timestamps, n_features)格式的数据，
    对每个sample计算速度、加速度、方差约束，然后合并所有结果用统一的分位数计算约束边界"""

    def __init__(self, data_3d: np.ndarray, k_sigma: float = 3.0):
        """
        :param data_3d: 三维numpy数组 (n_samples, n_timestamps, n_features)
        :param k_sigma: 缩放因子，通常取 3.0 (等效于99.7%置信度)
        """
        self.data_3d = data_3d
        self.window_size = 5
        self.k_sigma = k_sigma


    def _calc_speed(self, series: pd.Series):
        return series.diff().dropna()

    def _calc_accel(self, series: pd.Series):
        return series.diff().diff().dropna()

    def _calc_variance(self, series: pd.Series):
        return series.rolling(window=self.window_size).var().dropna()
    
    def _get_robust_bounds(self, data_list):
        """
        核心稳健算法：基于 MAD 计算边界
        """
        arr = np.array(data_list)
        if arr.size == 0:
            return 0.0, 0.0
        
        # 1. 计算中位数 (崩溃点 50%)
        med = np.median(arr)
        
        # 2. 计算 MAD
        mad = np.median(np.abs(arr - med))
        
        # 3. 转换为稳健标准差 (1.4826 是正态分布补偿系数)
        robust_std = mad * 1.4826
        
        # 4. 计算边界
        # 如果 robust_std 为 0 (说明数据太干净或全是重复值)，
        # 可以适当给一个微小的 epsilon 防止约束过死
        lower = med - self.k_sigma * robust_std
        upper = med + self.k_sigma * robust_std
        
        return lower, upper

    def mine_col_constraints_3d(self):
        """
        在三维数据上挖掘列约束
        
        返回:
            speed_constraints: {col: (min_speed, max_speed)}
            accel_constraints: {col: (min_accel, max_accel)}
            variance_constraints: {col: (min_local_var, max_local_var)}
        """
        n_samples, n_timestamps, n_features = self.data_3d.shape
        col_names = ["timestamp"] + [f"col_{i}" for i in range(1, n_features)]
        
        # 初始化聚合容器
        agg_data = {col: {"speed": [], "accel": [], "var": []} 
                    for col in col_names if col != "timestamp"}
        
        # 遍历样本提取特征
        for s_idx in range(n_samples):
            sample_df = pd.DataFrame(self.data_3d[s_idx], columns=col_names)
            
            for col in agg_data.keys():
                if col == "timestamp":  # 不对 timestamp 列计算约束
                    continue

                series = sample_df[col].astype(np.float64)
                
                # 提取特征并过滤无效值
                s = self._calc_speed(series)
                agg_data[col]["speed"].extend(s[np.isfinite(s)].tolist())
                
                a = self._calc_accel(series)
                agg_data[col]["accel"].extend(a[np.isfinite(a)].tolist())
                
                v = self._calc_variance(series)
                agg_data[col]["var"].extend(v[np.isfinite(v)].tolist())

        # 计算最终稳健约束
        speed_cons, accel_cons, var_cons = {}, {}, {}
        
        for col in agg_data.keys():
            speed_cons[col] = self._get_robust_bounds(agg_data[col]["speed"])
            accel_cons[col] = self._get_robust_bounds(agg_data[col]["accel"])
            var_cons[col] = self._get_robust_bounds(agg_data[col]["var"])
            
        return speed_cons, accel_cons, var_cons
        

    def mine_col_constraints_3d_per_sample(self):
        """
        在三维数据上为每个样本单独挖掘列约束
        
        返回:
            List of tuples, each tuple contains:
                speed_constraints: {col: (min_speed, max_speed)}
                accel_constraints: {col: (min_accel, max_accel)}
                variance_constraints: {col: (min_local_var, max_local_var)}
        """
        n_samples, _, n_features = self.data_3d.shape
        col_names = ["timestamp"] + [f"col_{i}" for i in range(1, n_features)]
        results = []

        for s_idx in range(n_samples):
            sample_df = pd.DataFrame(self.data_3d[s_idx], columns=col_names)
            s_speed, s_accel, s_var = {}, {}, {}
            
            for col in col_names:
                if col == "timestamp": continue
                
                series = sample_df[col].astype(np.float64)
                s_speed[col] = self._get_robust_bounds(self._calc_speed(series))
                s_accel[col] = self._get_robust_bounds(self._calc_accel(series))
                s_var[col] = self._get_robust_bounds(self._calc_variance(series))
                
            results.append((s_speed, s_accel, s_var))
            
        return results


class OrderDependencyMiner:
    """
    Order Dependency (OD) 约束挖掘器

    定义：如果属性 A 的取值排序决定了属性 B 的排序（单调关系），则 A → B 是一个 OD。
    例子：在时序数据中，如果时间戳递增，那么"库存量"也单调不减。
    """

    def __init__(self, df=None, method="kendall", threshold=0.9):
        """
        初始化 OD 挖掘器

        :param df: 数据（numpy数组或DataFrame）
        :param method: 检测方法 ('kendall', 'spearman')
        :param threshold: 阈值，相关性或违例比例判断依据
        """
        if isinstance(df, np.ndarray) and df.ndim == 2:
            column_names = [f"col_{i}" for i in range(df.shape[1])]
            df = pd.DataFrame(df, columns=column_names)
        self.df = df
        self.method = method
        self.threshold = threshold
        self.od_constraints = []

    def _check_monotonicity(self, series_a, series_b):
        """
        相关性检测（快速筛选候选 OD）
        """
        if self.method == "kendall":
            correlation = series_a.corr(series_b, method="kendall")
        elif self.method == "spearman":
            correlation = series_a.corr(series_b, method="spearman")
        else:
            raise ValueError("Method must be 'kendall' or 'spearman'")

        is_monotonic = abs(correlation) >= self.threshold
        return is_monotonic, correlation

    def _check_order_dependency(self, series_a, series_b):
        """
        排序检测（验证是否递增/递减单调）
        """
        clean_data = pd.DataFrame({"A": series_a, "B": series_b}).dropna()
        if len(clean_data) < 2:
            return False, "none", 0, 0

        # 按 A 排序，检查 B 的单调性
        sorted_data = clean_data.sort_values("A")
        b_values = sorted_data["B"].values

        # 递增与递减违例
        inc_violations = np.sum(np.diff(b_values) < 0)
        dec_violations = np.sum(np.diff(b_values) > 0)
        total_pairs = len(b_values) - 1

        inc_ratio = 1 - inc_violations / total_pairs  # 满足递增比例
        dec_ratio = 1 - dec_violations / total_pairs  # 满足递减比例

        # 判断是否满足约束
        if inc_ratio >= self.threshold:
            return True, "increasing", inc_ratio, inc_violations
        elif dec_ratio >= self.threshold:
            return True, "decreasing", dec_ratio, dec_violations
        else:
            return (
                False,
                "none",
                max(inc_ratio, dec_ratio),
                min(inc_violations, dec_violations),
            )

    def mine_order_dependencies(self):
        """
        挖掘所有属性之间的 OD 关系（仅当超过 threshold 比例满足 order 约束时保留）
        """
        columns = self.df.columns
        od_constraints = []

        for col_a in columns:
            for col_b in columns:
                if col_a == col_b:
                    continue

                # 相关性筛选（快速过滤）
                is_monotonic, correlation = self._check_monotonicity(
                    self.df[col_a], self.df[col_b]
                )

                # 排序检测
                is_dep, monotonicity, satisfy_ratio, violations = (
                    self._check_order_dependency(self.df[col_a], self.df[col_b])
                )

                # 仅当满足比例 ≥ 阈值时保留
                if is_dep and satisfy_ratio >= self.threshold:
                    constraint = {
                        "source": col_a,
                        "target": col_b,
                        "correlation": correlation,
                        "violations": violations,
                        "satisfy_ratio": satisfy_ratio,
                        "monotonicity": (
                            monotonicity
                            if monotonicity != "none"
                            else ("increasing" if correlation > 0 else "decreasing")
                        ),
                    }
                    od_constraints.append(constraint)

        self.od_constraints = od_constraints
        return od_constraints

    def mine_3d(self, data_3d: np.ndarray) -> List[Dict[str, Any]]:
        """
        在 3D 数据上挖掘 Order Dependencies。
        对每个 sample 单独挖掘（使用同样的 threshold），然后取那些在所有 sample
        上均满足（satisfy_ratio >= threshold）的 OD。
        """
        arr = np.asarray(data_3d)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        n_samples = arr.shape[0]
        n_features = arr.shape[2]
        candidates: Set[Tuple[str, str, str]] = set()
        candidate_meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # 在每个样本上挖掘候选 OD
        for i in range(n_samples):
            sample = arr[i]
            cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
            df_sample = pd.DataFrame(sample, columns=cols)

            miner = OrderDependencyMiner(df_sample, method=self.method, threshold=self.threshold)
            try:
                ods = miner.mine_order_dependencies()
            except Exception:
                ods = []

            for od in ods:
                key = (od["source"], od["target"], od.get("monotonicity", "unknown"))
                if key not in candidates:
                    candidates.add(key)
                    candidate_meta[key] = od

        # 验证候选在所有样本上是否成立
        final = []
        for key, meta in candidate_meta.items():
            src, tgt, mon = key
            ok = True
            for i in range(n_samples):
                sample = arr[i]
                cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
                df_sample = pd.DataFrame(sample, columns=cols)

                is_dep, monotonicity, satisfy_ratio, violations = self._check_order_dependency(df_sample[src], df_sample[tgt])
                if not is_dep or satisfy_ratio < self.threshold:
                    ok = False
                    break

            if ok:
                final.append(meta)

        self.od_constraints = final
        return final

    def mine_3d_per_sample(self, data_3d: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        在 3D 数据上为每个样本单独挖掘 Order Dependencies。
        """
        arr = np.asarray(data_3d)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        n_samples = arr.shape[0]
        n_features = arr.shape[2]
        sample_od_constraints: List[List[Dict[str, Any]]] = []

        # 在每个样本上挖掘 OD
        for i in range(n_samples):
            sample = arr[i]
            cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
            df_sample = pd.DataFrame(sample, columns=cols)

            miner = OrderDependencyMiner(df_sample, method=self.method, threshold=self.threshold)
            try:
                ods = miner.mine_order_dependencies()
            except Exception:
                ods = []
            sample_od_constraints.append(ods)

        return sample_od_constraints


class DenialDependencyMiner:
    def __init__(self, df=None, n_bins=3, strategy="fixed"):
        """
        :param df: 输入的多变量时序数据 (numpy数组或DataFrame)
        :param n_bins: 分箱数 (int 或 dict )
        :param strategy: 'fixed' 固定分箱, 'auto' 自适应
        """

        if isinstance(df, np.ndarray) and df.ndim == 2:
            column_names = [f"col_{i}" for i in range(df.shape[1])]
            df = pd.DataFrame(df, columns=column_names)
        self.df = df
        self.n_bins = n_bins
        self.strategy = strategy
        self.predicates = {}
        self.candidates = []
        self.top_k = 5

    def get_n_bins(self, col):
        """根据策略确定某个变量的分箱数"""
        if isinstance(self.n_bins, dict):
            return self.n_bins.get(col, 3)
        elif isinstance(self.n_bins, int):
            return self.n_bins
        elif self.strategy == "auto":
            N = len(self.df[col])
            iqr = np.percentile(self.df[col], 75) - np.percentile(self.df[col], 25)
            bin_width = 2 * iqr / (N ** (1 / 3) + 1e-6)
            n_bins = max(
                2, int((self.df[col].max() - self.df[col].min()) / (bin_width + 1e-6))
            )
            return min(n_bins, 10)
        else:
            return 3

    def discretize_variable(self, series, n_bins):
        """基于分位数对变量进行分桶"""
        bins = np.quantile(series, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 1e-6
        bins[-1] += 1e-6
        return bins

    def generate_predicates(self):
        """为所有数值列生成谓词"""
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                bins = self.discretize_variable(self.df[col], self.get_n_bins(col))
                preds = []
                for i in range(len(bins) - 1):
                    preds.append(f"{col} >= {bins[i]:.2f} and {col} < {bins[i+1]:.2f}")
                self.predicates[col] = preds
        return self.predicates

    def generate_candidates(self):
        """生成候选DD（只考虑跨变量组合）"""
        all_preds = []
        for col, preds in self.predicates.items():
            for p in preds:
                all_preds.append((col, p))

        self.candidates = []
        for (col1, p1), (col2, p2) in combinations(all_preds, 2):
            if col1 != col2:
                self.candidates.append(f"¬({p1} and {p2})")
        return self.candidates

    def validate_all(self, min_support=0.1, min_conf=0.9, sort_by="confidence"):
        """
        验证并筛选DD
        :param min_support: 最小覆盖率 (支持度阈值)
        :param min_conf: 最小置信度
        :param sort_by: 排序方式 "confidence" 或 "support"
        """
        results = []
        N = len(self.df)

        for c in self.candidates:
            # 候选格式: ¬(p1 and p2)
            inner = c.strip("¬()")
            # 小心分割，这里要按最后一个 ' and ' 分
            split_idx = inner.rfind(" and ")
            p1, p2 = inner[:split_idx], inner[split_idx + 5 :]

            try:
                subset = self.df.query(f"{p1} and {p2}")
            except Exception as e:
                # 如果 query 失败，跳过
                continue

            # 支持度: p1∧p2 在全局数据中的比例
            support = len(subset) / N

            # 计算相关记录 (p1 ∨ p2)
            related = self.df.query(f"{p1} or {p2}")
            if len(related) > 0:
                conf = 1 - len(subset) / len(related)
            else:
                conf = 1.0  # 没有相关记录时，认为完全满足

            if support < min_support:
                continue
            if conf < min_conf:
                continue

            results.append({"dd": c, "support": support, "confidence": conf})

        # 排序
        if sort_by == "confidence":
            results.sort(key=lambda x: (-x["confidence"], -x["support"]))
        elif sort_by == "support":
            results.sort(key=lambda x: (-x["support"], -x["confidence"]))

        results = results[: self.top_k]

        return results

    def mine_3d(self, data_3d: np.ndarray, min_support=0.1, min_conf=0.9) -> List[Dict[str, Any]]:
        """
        在 3D 数据上挖掘否定依赖（DD）。
        方案：对每个 sample 单独生成谓词、候选并做初步验证，收集所有候选（字符串形式），
        然后对每个候选在所有 sample 上进行验证（使用同一条数值阈值表达式在每个 sample 上执行 query），
        只有在所有 sample 上均满足 min_support 和 min_conf 的候选才被保留。
        """
        arr = np.asarray(data_3d)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        n_samples = arr.shape[0]
        n_features = arr.shape[2]
        all_candidates: Set[str] = set()
        candidate_meta: Dict[str, Dict[str, Any]] = {}

        # 在每个 sample 上生成候选
        for i in range(n_samples):
            sample = arr[i]
            cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
            df_sample = pd.DataFrame(sample, columns=cols)

            miner = DenialDependencyMiner(df_sample, n_bins=self.n_bins, strategy=self.strategy)
            try:
                miner.generate_predicates()
                miner.generate_candidates()
                validated = miner.validate_all(min_support=min_support, min_conf=min_conf, sort_by="confidence")
            except Exception:
                validated = []

            # 收集候选字符串（使用未必验证通过的所有候选以保障广泛性）
            for (col, preds) in miner.predicates.items():
                for p in preds:
                    # we will combine cross-variable pairs; but miner.generate_candidates already built combos
                    pass

            # add validated ones as seeds
            for dd in validated:
                all_candidates.add(dd["dd"]) if isinstance(dd, dict) and "dd" in dd else None

            # also include raw candidates produced (to enlarge candidate set)
            for c in miner.candidates:
                all_candidates.add(c)

        # 在所有样本上验证候选
        final_results: List[Dict[str, Any]] = []
        for c in all_candidates:
            inner = c.strip("¬()")
            split_idx = inner.rfind(" and ")
            if split_idx == -1:
                continue
            p1, p2 = inner[:split_idx], inner[split_idx + 5 :]

            ok = True
            supports = []
            confs = []

            for i in range(n_samples):
                sample = arr[i]
                cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
                df_sample = pd.DataFrame(sample, columns=cols)

                try:
                    subset = df_sample.query(f"{p1} and {p2}")
                except Exception:
                    ok = False
                    break

                support = len(subset) / max(1, len(df_sample))
                try:
                    related = df_sample.query(f"{p1} or {p2}")
                except Exception:
                    related = df_sample

                if len(related) > 0:
                    conf = 1 - len(subset) / len(related)
                else:
                    conf = 1.0

                supports.append(support)
                confs.append(conf)

                if support < min_support or conf < min_conf:
                    ok = False
                    break

            if ok:
                final_results.append({"dd": c, "support": float(np.mean(supports)), "confidence": float(np.mean(confs))})

        # 排序并截断
        final_results.sort(key=lambda x: (-x["confidence"], -x["support"]))
        return final_results[: self.top_k]

    def mine_3d_per_sample(self, data_3d: np.ndarray, min_support=0.1, min_conf=0.9) -> List[List[Dict[str, Any]]]:
        """
        在 3D 数据上为每个样本单独挖掘否定依赖（DD）。
        """
        arr = np.asarray(data_3d)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])

        n_samples = arr.shape[0]
        n_features = arr.shape[2]
        sample_dd_constraints: List[List[Dict[str, Any]]] = []

        # 在每个 sample 上生成候选
        for i in range(n_samples):
            sample = arr[i]
            cols = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
            df_sample = pd.DataFrame(sample, columns=cols)

            miner = DenialDependencyMiner(df_sample, n_bins=self.n_bins, strategy=self.strategy)
            try:
                miner.generate_predicates()
                miner.generate_candidates()
                validated = miner.validate_all(min_support=min_support, min_conf=min_conf, sort_by="confidence")
            except Exception:
                validated = []
            sample_dd_constraints.append(validated)

        return sample_dd_constraints


def mine_all_constraints(
    data_3d: np.ndarray,
    degree=2,
    attr_num=3,
    outlier_rate=0.1,
    confidence_threshold=0.95,
    window=10,
    n_bins=3,
    strategy="auto",
    min_support=0.05,
    min_conf=0.9,
):
    
    # 支持传入 2D (single sample DataFrame/ndarray) 或 3D numpy array
    arr = np.asarray(data_3d)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    elif arr.ndim != 3:
        raise AssertionError("Input data must be 2D (single sample) or 3D numpy array (n_samples, n_timestamps, n_features)")

    n_features = arr.shape[2]
    col_names = ["timestamp"] + [f"col_{i}" for i in range(1, n_features)]
    
    # Filter out non-numeric columns
    numeric_cols = []
    numeric_col_names = []
    
    for i in range(n_features):
        # Check if column is numeric by trying to convert a sample to float
        try:
            # Try converting first few values of the column to check if numeric
            sample_vals = arr[0, :min(5, arr.shape[1]), i]  # Check first 5 values of first sample
            is_numeric = True
            for val in sample_vals:
                if pd.notna(val):  # Skip NaN values
                    float(val)  # Try converting to float
            # If successful, include this column
            numeric_cols.append(i)
            numeric_col_names.append(col_names[i])
        except (ValueError, TypeError):
            # Skip non-numeric columns
            continue
    
    # If we have numeric data, create new array with only numeric columns
    if numeric_cols:
        numeric_data_3d = arr[:, :, numeric_cols]
        # print(f"发现 {len(numeric_cols)} 个数值型列: {numeric_col_names}")
        # print(f"数值型数据形状: {numeric_data_3d.shape}")
        
        # 检查数据中是否有变化
        for i, col_name in enumerate(numeric_col_names):
            col_data = numeric_data_3d[0, :, i].astype(float)  # 第一个样本
            finite_data = col_data[np.isfinite(col_data)]
            if len(finite_data) > 1:
                diff = np.diff(finite_data)
                # if len(diff) > 0:
                    # print(f"列 {col_name} 数据范围: [{np.min(finite_data):.6f}, {np.max(finite_data):.6f}], 差分范围: [{np.min(diff):.6f}, {np.max(diff):.6f}]")

    else:
        # If no numeric columns, create an empty array with the right shape
        numeric_data_3d = np.empty((arr.shape[0], arr.shape[1], 0))
        print("未发现数值型列")
    
    # Only mine constraints if we have numeric data
    if numeric_data_3d.shape[2] > 0:
        # print(f"使用形状为 {numeric_data_3d.shape} 的数据进行约束挖掘")
        
        # ROW constraints: mine candidates from each sample then validate across samples
        row_miner = RowConstraintMiner(degree=degree, max_features=attr_num)
        row_constraints = row_miner.mine_3d(numeric_data_3d)

        # COLUMN constraints (speed/accel/variance)
        col_miner = ColConstraintMiner(numeric_data_3d)
        (
            speed_constraints,
            accel_constraints,
            variance_constraints,
        ) = col_miner.mine_col_constraints_3d()

        # ORDER DEPENDENCY
        od_miner = OrderDependencyMiner(None, method="kendall", threshold=confidence_threshold)
        od_constraints = od_miner.mine_3d(numeric_data_3d)

        # DENIAL DEPENDENCY
        dd_miner = DenialDependencyMiner(None, n_bins=n_bins, strategy=strategy)
        dd_constraints = dd_miner.mine_3d(numeric_data_3d, min_support=min_support, min_conf=min_conf)
    else:
        # Return empty constraints if no numeric data
        print("由于没有数值型数据，返回空约束")
        row_constraints = []
        speed_constraints = {}
        accel_constraints = {}
        variance_constraints = {}
        od_constraints = []
        dd_constraints = []

    return {
        "row_constraints": row_constraints,
        "speed_constraints": speed_constraints,
        "accel_constraints": accel_constraints,
        "variance_constraints": variance_constraints,
        "od_constraints": od_constraints,
        "dd_constraints": dd_constraints,
    }


def mine_all_constraints_per_sample(
    data_3d: np.ndarray,
    degree=2,
    attr_num=2,
    outlier_rate=0.1,
    confidence_threshold=0.95,
    window=10,
    n_bins=3,
    strategy="auto",
    min_support=0.05,
    min_conf=0.9,
):
    """
    为每个样本分别进行约束挖掘，并将每个样本对应的约束放在一起返回
    明确使用各miner类的_per_sample方法来体现每个样本独立挖掘约束的意图
    """
    # 支持传入 2D (single sample DataFrame/ndarray) 或 3D numpy array
    arr = np.asarray(data_3d)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    elif arr.ndim != 3:
        raise AssertionError("Input data must be 2D (single sample) or 3D numpy array (n_samples, n_timestamps, n_features)")

    n_samples = arr.shape[0]
    n_features = arr.shape[2]
    col_names = ["timestamp"] + [f"col_{i}" for i in range(1, n_features)]
    
    # Filter out non-numeric columns
    numeric_cols = []
    numeric_col_names = []
    
    for i in range(n_features):
        # Check if column is numeric by trying to convert a sample to float
        try:
            # Try converting first few values of the column to check if numeric
            sample_vals = arr[0, :min(5, arr.shape[1]), i]  # Check first 5 values of first sample
            is_numeric = True
            for val in sample_vals:
                if pd.notna(val):  # Skip NaN values
                    float(val)  # Try converting to float
            # If successful, include this column
            numeric_cols.append(i)
            numeric_col_names.append(col_names[i])
        except (ValueError, TypeError):
            # Skip non-numeric columns
            continue
    
    # If we have numeric data, create new array with only numeric columns
    if numeric_cols:
        numeric_data_3d = arr[:, :, numeric_cols]
        # print(f"发现 {len(numeric_cols)} 个数值型列: {numeric_col_names}")
        # print(f"数值型数据形状: {numeric_data_3d.shape}")
    else:
        # If no numeric columns, create an empty array with the right shape
        numeric_data_3d = np.empty((arr.shape[0], arr.shape[1], 0))
        print("未发现数值型列")
    
    # Handle outlier_rate as either a single value or a list
    if isinstance(outlier_rate, (list, tuple, np.ndarray)):
        if len(outlier_rate) != n_samples:
            raise ValueError(f"outlier_rate列表长度({len(outlier_rate)})必须等于样本数({n_samples})")
        outlier_rates = outlier_rate
    else:
        # If outlier_rate is a single value, use it for all samples
        outlier_rates = [outlier_rate] * n_samples
    
    # Mine constraints per sample using the per-sample methods
    print(f"为 {n_samples} 个样本分别挖掘约束...")
    
    # ROW constraints per sample
    row_miner = RowConstraintMiner(degree=degree, max_features=attr_num)
    row_constraints_per_sample = row_miner.mine_3d_per_sample(numeric_data_3d)

    # COLUMN constraints per sample - 为每个样本使用对应的异常率
    col_constraints_per_sample = []
    for sample_idx in range(n_samples):
        # 为当前样本创建带有对应异常率的ColConstraintMiner
        col_miner = ColConstraintMiner(numeric_data_3d[sample_idx:sample_idx+1])
        # 挖掘当前样本的列约束
        sample_col_constraints = col_miner.mine_col_constraints_3d_per_sample()
        col_constraints_per_sample.extend(sample_col_constraints)

    # ORDER DEPENDENCY constraints per sample
    od_miner = OrderDependencyMiner(method="kendall", threshold=confidence_threshold)
    od_constraints_per_sample = od_miner.mine_3d_per_sample(numeric_data_3d)

    # DENIAL DEPENDENCY constraints per sample
    dd_miner = DenialDependencyMiner(n_bins=n_bins, strategy=strategy)
    dd_constraints_per_sample = dd_miner.mine_3d_per_sample(numeric_data_3d, min_support=min_support, min_conf=min_conf)
    
    # Combine all constraints per sample
    sample_constraints = []
    for sample_idx in range(n_samples):
        sample_constraints.append({
            "sample_index": sample_idx,
            "row_constraints": row_constraints_per_sample[sample_idx] if sample_idx < len(row_constraints_per_sample) else [],
            "speed_constraints": col_constraints_per_sample[sample_idx][0] if sample_idx < len(col_constraints_per_sample) else {},
            "accel_constraints": col_constraints_per_sample[sample_idx][1] if sample_idx < len(col_constraints_per_sample) else {},
            "variance_constraints": col_constraints_per_sample[sample_idx][2] if sample_idx < len(col_constraints_per_sample) else {},
            "od_constraints": od_constraints_per_sample[sample_idx] if sample_idx < len(od_constraints_per_sample) else [],
            "dd_constraints": dd_constraints_per_sample[sample_idx] if sample_idx < len(dd_constraints_per_sample) else [],
        })
    
    return sample_constraints


def constraint_report(constraints):
    print("\n行约束:")
    if constraints["row_constraints"]:
        for c in constraints["row_constraints"]:
            print(c["formula"], "| loss:", round(c["loss_mse"], 6), "| residual_std:", round(c["residual_std"], 6))

    # 打印列约束
    print("\n速度约束:")
    if constraints["speed_constraints"]:
        for col, (min_val, max_val) in constraints["speed_constraints"].items():
            print(f"{col}: [{min_val:.6f}, {max_val:.6f}]")

    print("\n加速度约束:")
    if constraints["accel_constraints"]:
        for col, (min_val, max_val) in constraints["accel_constraints"].items():
            print(f"{col}: [{min_val:.6f}, {max_val:.6f}]")

    print("\n方差约束:")
    if constraints["variance_constraints"]:
        for col, (min_val, max_val) in constraints["variance_constraints"].items():
            print(f"{col}: [{min_val:.6f}, {max_val:.6f}]")



    print("\n=== 单调约束 ===")
    if constraints["od_constraints"]:
        for constraint in constraints["od_constraints"]:
            print(f"  {constraint['source']} → {constraint['target']}")
            print(f"    相关性: {constraint['correlation']:.6f}")
            print(f"    单调性: {constraint['monotonicity']}")
            print()



    print("\n=== 否定约束 ===")
    if constraints["dd_constraints"]:
        for constraint in constraints["dd_constraints"]:
            print(
                constraint["dd"],
                "support:",
                f"{constraint['support']:.6f}",
                "confidence:",
                f"{constraint['confidence']:.6f}",
            )




# 示例
if __name__ == "__main__":
    # 设置随机种子以便复现结果（可选）
    import random
    random.seed(42)

    # 构造三维示例数据，类似于原始的data结构
    # 创建两个样本，每个样本都类似于原始的data
    
    # 第一个样本
    data1 = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    }
    
    # 第二个样本（在原始数据基础上添加一些变化）
    data2 = data1.copy()
    
    # 转换为numpy数组
    sample1 = np.array([data1["A"], data1["B"], data1["C"], data1["D"]]).T
    sample2 = np.array([data2["A"], data2["B"], data2["C"], data2["D"]]).T
    
    # 合并为三维数组 (2, 10, 4) -> (n_samples, n_timestamps, n_features)
    data_3d = np.array([sample1, sample2])
    
    print("三维数据形状:", data_3d.shape)
    print("第一个样本:")
    print(data_3d[0])
    print("第二个样本:")
    print(data_3d[1])

    # 使用三维数据进行约束挖掘
    constraints = mine_all_constraints(
        data_3d,
        degree=3,
        attr_num=4,
        window=20,
        strategy="auto",
        min_support=0.1,
        min_conf=0.95,
        confidence_threshold=0.7
    )

    constraint_report(constraints)