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

# class RowConstraintMiner:
#     """
#     行约束挖掘器（PolynomialFeatures + Ridge）
#     constraints 直接包含公式字符串
#     """

#     def __init__(self, df, degree=2, alpha=1.0, tol=0.05, loss_threshold=0.01):
#         """
#         :param df: 待挖掘的数据（数值型）
#         :param degree: 多项式特征的最高阶
#         :param alpha: Ridge 回归正则化系数
#         :param tol: 约束容差（用于残差容忍区间半宽）
#         :param loss_threshold: 约束损失阈值（MSE）
#         """
#         self.df = df
#         self.degree = degree
#         self.alpha = alpha
#         self.tol = tol
#         self.loss_threshold = loss_threshold
#         self.constraints = []  # 存储字典：包含公式字符串和其他信息
#         self.top_k = 5

#     def row_miner(self, attr_num=2):
#         """
#         挖掘行约束
#         :param attr_num: 每条约束参与的变量总数（包括目标变量）
#         :return: top-k 低损失约束列表
#         """
#         df = self.df.copy()
#         features = df.columns
#         constraints = []

#         for combo in combinations(features, attr_num):
#             for target in combo:
#                 X_cols = [c for c in combo if c != target]
#                 if len(X_cols) == 0:
#                     continue

#                 X = df[X_cols].values
#                 y = df[target].values

#                 # 多项式特征生成
#                 poly = PolynomialFeatures(degree=self.degree, include_bias=False)
#                 X_poly = poly.fit_transform(X)
#                 feature_names = poly.get_feature_names_out(X_cols)

#                 # Ridge 回归
#                 model = Ridge(alpha=self.alpha, fit_intercept=True)
#                 model.fit(X_poly, y)
#                 y_pred = model.predict(X_poly)
#                 loss = mean_squared_error(y, y_pred)

#                 # 只保留低损失的模型
#                 if loss > self.loss_threshold:
#                     continue

#                 # 残差容忍区间：对称 around 0
#                 residuals = y - y_pred
#                 # 使用对称容忍区间：[-tol, +tol]，或基于标准差动态设定
#                 # 这里采用固定容忍（用户传入的 tol 作为半宽）
#                 lower = -self.tol
#                 upper = self.tol

#                 # 构建表达式字符串：处理正负号
#                 terms = []
#                 for coef, name in zip(model.coef_, feature_names):
#                     if abs(coef) < 1e-4:
#                         continue
#                     if coef >= 0:
#                         terms.append(f"+ {coef:.3f}*{name}")
#                     else:
#                         terms.append(f"- {abs(coef):.3f}*{name}")

#                 # 添加截距项
#                 if abs(model.intercept_) > 1e-6:
#                     if model.intercept_ >= 0:
#                         terms.append(f"+ {model.intercept_:.3f}")
#                     else:
#                         terms.append(f"- {abs(model.intercept_):.3f}")

#                 # 去掉首项的 '+' 号
#                 if terms:
#                     first = terms[0]
#                     if first.startswith("+ "):
#                         terms[0] = first[2:]

#                 # 构造公式：lower <= target - prediction <= upper
#                 pred_expr = " ".join(terms) if terms else "0.0"
#                 constraint_str = f"{lower:.3f} <= {target} - ({pred_expr}) <= {upper:.3f}"

#                 constraints.append({
#                     "formula": constraint_str,
#                     "target": target,
#                     "features": X_cols,
#                     "coef": dict(zip(feature_names, model.coef_)),  # 所有系数（含0）
#                     "intercept": model.intercept_,
#                     "lower": lower,
#                     "upper": upper,
#                     "loss": loss,
#                 })

#         # 按损失排序，取 top-k
#         constraints_sorted = sorted(constraints, key=lambda x: x["loss"])
#         self.constraints = constraints_sorted[:self.top_k]

#         return self.constraints


class RowConstraintMiner:
    """
        优化版行约束挖掘器：专注于挖掘精确、稀疏的近似函数关系。
        使用 OLS (普通最小二乘法) 替代 Lasso 来保证对完美线性关系的精确捕获，
        并通过稀疏化和残差分析来筛选结果。
        """

    def __init__(
        self,
        df: pd.DataFrame,
        degree: int = 2,
        max_features: int = 3,
        coverage: float = 0.9,
        max_residual_std: float = 0.05,  # 替代 min_support，更关注模型的拟合优度
        min_r2: float = 0.99,  # 最小 R-squared
    ):
        """
        :param df: 输入数据框（自动筛选数值列）
        :param degree: 多项式最高次数
        :param max_features: 最大输入变量数
        :param coverage: 残差容忍区间覆盖的比例 (用于定义 [lower, upper])
        :param max_residual_std: 约束成立时，残差的最大标准差（用于筛选模型）
        :param min_r2: 最小的 R-squared 值（用于筛选模型）
        """
        # 筛选数值列并清理
        self.df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
        self.degree = degree
        self.max_features = max_features
        self.coverage = coverage
        self.max_residual_std = max_residual_std
        self.min_r2 = min_r2
        self.constraints: List[Dict[str, Any]] = []

    def _create_poly(self, X: np.ndarray, cols: List[str]):
        """生成多项式特征并返回特征名"""
        # 仅为特征工程使用，不标准化原始数据
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(cols)
        return X_poly, feature_names, poly

    def _natural_sort_key(self, s: str) -> List:
        """自然排序键：确保 x1, x2, x10 正确排序"""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    def _format(self, coef_dict: Dict[str, float], intercept: float) -> str:
        """格式化为数学表达式，将系数保留 4 位小数，使之更精确"""
        if not coef_dict and abs(intercept) < 1e-4:
             return "0.000"

        # 按自然顺序排序变量名
        sorted_items = sorted(coef_dict.items(), key=lambda x: self._natural_sort_key(x[0]))
        terms = []

        # 1. 处理系数项
        for name, c in sorted_items:
            # 系数绝对值小于 1e-4 的项被忽略，实现了隐式稀疏化
            if abs(c) < 1e-4: 
                continue 
            
            sign = "+" if c >= 0 else "-"
            # 始终将系数放在前面
            terms.append(f"{sign} {abs(c):.4f}*{name}")
            
        expr = " ".join(terms).strip()

        # 2. 处理截距
        intercept_str = ""
        if abs(intercept) >= 1e-4:
            if expr:
                intercept_str = f"{intercept:.4f} + " if intercept >= 0 else f"{intercept:.4f} "
            else:
                return f"{intercept:.4f}" # 只有截距
        
        # 3. 组合表达式
        full_expr = f"{intercept_str}{expr}".strip()
        
        # 清理开头的 + 号
        if full_expr.startswith("+"):
            full_expr = full_expr[2:]

        return full_expr

    def _get_constraint_key(self, target: str, coef_dict: Dict[str, float], intercept: float) -> Tuple:
        """
        生成约束的唯一键（用于去重）
        将系数四舍五入到一定精度后构建元组
        """
        # 只保留系数大于 1e-4 的项
        rounded_coef = tuple(sorted(
            (k, round(v, 6)) for k, v in coef_dict.items() if abs(v) >= 1e-4
        ))
        rounded_intercept = round(intercept, 6)
        return (target, rounded_coef, rounded_intercept)

    def row_miner(self) -> List[Dict[str, Any]]:
        """
        挖掘行级约束，采用 OLS + 筛选策略
        """
        df = self.df
        columns = df.columns.tolist()
        constraints = []
        seen: Set[Tuple] = set()

        # 遍历每个变量作为目标
        for target in columns:
            candidates = [col for col in columns if col != target]

            # 尝试不同数量的输入变量
            for n in range(1, min(self.max_features, len(candidates)) + 1):
                for feat_combo in combinations(candidates, n):
                    X = df[list(feat_combo)].values
                    y = df[target].values
                    
                    # 1. 特征工程
                    try:
                        X_poly, feature_names, _ = self._create_poly(X, list(feat_combo))
                    except ValueError:
                        continue 

                    # 2. 模型拟合 (使用 OLS)
                    # OLS 寻找最优解，不会像 Lasso 那样强制稀疏（稀疏性通过后续的筛选和格式化实现）
                    model = LinearRegression(fit_intercept=True)
                    try:
                         # 使用 SVD 求解，在一般情况下比 Lasso 稳定且精确
                        model.fit(X_poly, y)
                    except ValueError:
                        continue 

                    pred = model.predict(X_poly)
                    residuals = y - pred

                    # 3. 约束评估与筛选
                    r2 = model.score(X_poly, y)
                    loss_mse = mean_squared_error(y, pred)
                    residual_std = np.std(residuals)

                    # 筛选条件：R^2 必须很高，且残差标准差必须很低
                    if r2 < self.min_r2 or residual_std > self.max_residual_std:
                        continue
                    
                    # 4. 稀疏化和去重 (隐式稀疏化在 _format 和 _get_constraint_key 中实现)
                    
                    # 将所有多项式特征项的系数都纳入考虑
                    coef_dict = {name: c for name, c in zip(feature_names, model.coef_)}

                    # 跳过空模型 (尽管 OLS 很难产生空模型，但仍保留逻辑)
                    if not coef_dict and abs(model.intercept_) < 1e-4:
                        continue

                    key = self._get_constraint_key(target, coef_dict, model.intercept_)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    # 5. 残差容忍区间确定
                    tail = (1 - self.coverage) / 2
                    lower = np.quantile(residuals, tail)
                    upper = np.quantile(residuals, 1 - tail)
                    
                    # 计算支持度 (在这个框架下，支持度应该接近 1.0)
                    satisfied = (residuals >= lower) & (residuals <= upper)
                    support = np.mean(satisfied)

                    # 6. 格式化和保存
                    expr = self._format(coef_dict, model.intercept_)
                    
                    # 仅保留在格式化后非空且不为纯截距的模型
                    if expr.strip() == "":
                        continue
                        
                    formula = f"{target} = {expr} | residual ∈ [{lower:.4f}, {upper:.4f}]"

                    constraints.append({
                        "formula": formula,
                        "target": target,
                        "features": list(feat_combo),
                        "model_terms": coef_dict,
                        "intercept": model.intercept_,
                        "lower": lower,
                        "upper": upper,
                        "support": support,
                        "R2": r2,
                        "loss_mse": loss_mse,
                        "residual_std": residual_std,
                    })

        # 按 R2 降序，MSE 升序排序
        constraints.sort(key=lambda x: (-x["R2"], x["loss_mse"]))
        self.constraints = constraints
        return constraints


class ColConstraintMiner:
    """列约束挖掘器：自动从时间序列中提取速度、加速度、幅值、方差等约束"""

    def __init__(self, df: pd.DataFrame, q_low=0.05, q_high=0.95):
        self.df = df
        self.q_low = q_low
        self.q_high = q_high
        self.window_size = 5

    def _is_numeric(self, series: pd.Series) -> bool:
        """判断是否为数值型列"""
        return pd.api.types.is_numeric_dtype(series)

    def _calc_speed(self, series: pd.Series, window=None):
        """计算一阶差分（变化率）的分位数约束"""
        diffs = series.diff(periods=1)
        diffs = diffs.dropna()  # 避免 fillna 引入偏差
        if diffs.empty:
            return (0, 0)
        return (diffs.quantile(self.q_low), diffs.quantile(self.q_high))

    def _calc_accel(self, series: pd.Series, window=None):
        """计算二阶差分（加速度）的分位数约束"""
        diffs = series.diff(periods=1).diff(periods=1)
        diffs = diffs.dropna()
        if diffs.empty:
            return (0, 0)
        return (diffs.quantile(self.q_low), diffs.quantile(self.q_high))

    # def _calc_amplitude(self, series: pd.Series):
    #     """计算幅值范围（原始值的分位数）"""
    #     valid = series.dropna()
    #     if valid.empty:
    #         return (0, 0)
    #     return (valid.quantile(self.q_low), valid.quantile(self.q_high))

    def _calc_variance(self, series: pd.Series):
        """
        计算局部方差约束。
        返回：(min_allowed_variance, max_allowed_variance)
        实际常用的是上界，但这里返回区间以供灵活使用。
        """
        w=self.window_size
        rolling_var = series.rolling(window=w).var()
        rolling_var = rolling_var.dropna()
        if rolling_var.empty:
            var_val = np.var(series.dropna()) or 0
            return (var_val, var_val)
        
        low_bound = 0
        high_bound = rolling_var.quantile(self.q_high)

        return (low_bound, high_bound)

    def mine_col_constraints(
        self,
        time_col="timestamp",
    ):
        """
        挖掘所有数值列的四类约束
        
        返回:
            speed_constraints: {col: (min_speed, max_speed)}
            accel_constraints: {col: (min_accel, max_accel)}
            variance_constraints: {col: (min_local_var, max_local_var)}  # 注意解释
            amplitude_constraints: {col: (min_val, max_val)}
        """
        speed_constraints = {}
        accel_constraints = {}
        variance_constraints = {}
        # amplitude_constraints = {}

        for col in self.df.columns:
            if col == time_col or not self._is_numeric(self.df[col]):
                continue

            series = self.df[col]
            speed_constraints[col] = self._calc_speed(series)
            accel_constraints[col] = self._calc_accel(series)
            variance_constraints[col] = self._calc_variance(series)
            # amplitude_constraints[col] = self._calc_amplitude(series)

        return (
            speed_constraints,
            accel_constraints,
            variance_constraints,
            # amplitude_constraints,
        )


class OrderDependencyMiner:
    """
    Order Dependency (OD) 约束挖掘器

    定义：如果属性 A 的取值排序决定了属性 B 的排序（单调关系），则 A → B 是一个 OD。
    例子：在时序数据中，如果时间戳递增，那么"库存量"也单调不减。
    """

    def __init__(self, df, method="kendall", threshold=0.9):
        """
        初始化 OD 挖掘器

        :param df: 数据框
        :param method: 检测方法 ('kendall', 'spearman')
        :param threshold: 阈值，相关性或违例比例判断依据
        """
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


class SequentialDependencyMiner:
    def __init__(self, df, confidence_threshold=0.95, min_len=5):
        """
        :param df: 输入 DataFrame
        :param confidence_threshold: 判断 N 单调的置信度阈值
        :param min_len: 最小子序列长度
        """
        self.df = df.copy()
        self.confidence_threshold = confidence_threshold
        self.min_len = min_len
        self.results = []
        self.time_col = "timestamp"

    def _find_monotonic_segments(self, s):
        """
        给定一列 s，找到单调递增或递减的子序列区间 (start, end, direction)
        """
        idx = s.index.to_numpy()
        vals = s.to_numpy()

        if len(vals) < self.min_len:
            return []

        segments = []
        start = 0
        # 当前趋势（1=递增, -1=递减, 0=未定）
        trend = 0

        for i in range(1, len(vals)):
            diff = vals[i] - vals[i - 1]
            new_trend = 0
            if diff > 0:
                new_trend = 1
            elif diff < 0:
                new_trend = -1

            if trend == 0:
                trend = new_trend if new_trend != 0 else 0
            elif new_trend != 0 and new_trend != trend:
                # 趋势反转 -> 结束前一段
                if i - start >= self.min_len:
                    segments.append((idx[start], idx[i - 1], "increasing" if trend == 1 else "decreasing"))
                start = i - 1
                trend = new_trend

        # 收尾
        if len(vals) - start >= self.min_len and trend != 0:
            segments.append((idx[start], idx[len(vals) - 1], "increasing" if trend == 1 else "decreasing"))

        return segments

    def _check_monotonic_N(self, df_seg, N):
        """
        检查在 df_seg 里 N 是否单调，返回方向和置信度
        """
        delta = df_seg[N].diff().dropna().to_numpy()
        if len(delta) == 0:
            return None

        pos_ratio = np.mean(delta >= 0)
        neg_ratio = np.mean(delta <= 0)

        if pos_ratio >= self.confidence_threshold:
            return {"direction": "non-decreasing", "confidence": pos_ratio}
        elif neg_ratio >= self.confidence_threshold:
            return {"direction": "non-increasing", "confidence": neg_ratio}
        else:
            return None

    def mine(self, M_cols=None, N_cols=None):
        """
        主函数：对每个 M 的单调子序列，检测所有 N 的单调性
        """
        self.results = []
        if M_cols is None:
            M_cols = self.df.select_dtypes(include=[np.number, "datetime"]).columns.tolist()
        if N_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            N_cols = [c for c in numeric_cols if c.lower() not in ("timestamp", "time")]

        for M in M_cols:
            segments = self._find_monotonic_segments(self.df[M].dropna())
            for seg_start, seg_end, m_dir in segments:
                df_seg = self.df.loc[seg_start:seg_end]
                for N in N_cols:
                    res = self._check_monotonic_N(df_seg, N)
                    if res:
                        self.results.append({
                            "M": M,
                            "M_direction": m_dir,
                            "N": N,
                            "N_direction": res["direction"],
                            "segment": (seg_start, seg_end),
                            "confidence": res["confidence"],
                            "length": len(df_seg)
                        })
        return self.results


class DenialDependencyMiner:
    def __init__(self, df, n_bins, strategy="fixed"):
        """
        :param df: 输入的多变量时序数据 (DataFrame)
        :param n_bins: 分箱数 (int 或 dict )
        :param strategy: 'fixed' 固定分箱, 'auto' 自适应
        """
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


class TrendDependencyMiner:
    def __init__(self, df, window_size=1, confidence_threshold=0.8):
        """
        :param df: 输入 DataFrame
        :param window_size: 滑动窗口均值平滑大小，默认为1（不平滑）
        :param confidence_threshold: 正向或反向同步率阈值，超过该值则认为存在依赖
        """
        self.df = df.copy()
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.dependencies = []

    def _window_smooth(self, series):
        """滑动窗口均值平滑"""
        return series.rolling(window=self.window_size, min_periods=1).mean().values

    def _compute_trend_sync(self, M_vals, N_vals):
        """计算正向/反向趋势同步率"""
        delta_M = np.diff(M_vals)
        delta_N = np.diff(N_vals)
        mask = (delta_M != 0) & (delta_N != 0)
        if np.sum(mask) == 0:
            return 0.0, 0.0
        signs_M = np.sign(delta_M[mask])
        signs_N = np.sign(delta_N[mask])
        positive_sync = np.mean(signs_M == signs_N)
        negative_sync = np.mean(signs_M == -signs_N)
        return positive_sync, negative_sync

    def mine(self, M_cols=None, N_cols=None, exclude_cols="timestamp"):
        """
        挖掘变量之间的趋势依赖约束
        :param M_cols: 可选的M列列表
        :param N_cols: 可选的N列列表
        :param exclude_cols: 要排除的列列表，例如 ['timestamp']
        :return: list of dict，每个 dict 表示一个依赖约束
        """
        self.dependencies = []

        if exclude_cols is None:
            exclude_cols = []

        # 默认候选列：数值列，排除指定列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if M_cols is None:
            M_cols = numeric_cols
        if N_cols is None:
            N_cols = numeric_cols

        for M in M_cols:
            for N in N_cols:
                if M == N:
                    continue
                M_smooth = self._window_smooth(self.df[M])
                N_smooth = self._window_smooth(self.df[N])

                pos_sync, neg_sync = self._compute_trend_sync(M_smooth, N_smooth)

                if pos_sync >= self.confidence_threshold:
                    self.dependencies.append({
                        "M": M,
                        "N": N,
                        "trend_type": "positive",
                        "trend_confidence": pos_sync
                    })
                elif neg_sync >= self.confidence_threshold:
                    self.dependencies.append({
                        "M": M,
                        "N": N,
                        "trend_type": "negative",
                        "trend_confidence": neg_sync
                    })

        return self.dependencies

def mine_all_constraints(
    df,
    degree=3,
    attr_num=3,
    outlier_rate=0.1,
    confidence_threshold=0.95,
    window=10,
    n_bins=3,
    strategy="auto",
    min_support=0.05,
    min_conf=0.9,
):
    row_miner = RowConstraintMiner(df, degree)
    # row_constraints = row_miner.row_miner(attr_num)
    row_constraints = row_miner.row_miner()

    q_low = outlier_rate / 4
    q_high = 1 - outlier_rate / 4
    col_miner = ColConstraintMiner(df, q_low=q_low, q_high=q_high)
    (
        speed_constraints,
        accel_constraints,
        variance_constraints,
        # amplitude_constraints,
    ) = col_miner.mine_col_constraints()

    od_miner = OrderDependencyMiner(df)
    od_constraints = od_miner.mine_order_dependencies()

    # # 挖掘顺序依赖约束
    # sd_miner = SequentialDependencyMiner(df, confidence_threshold, window)
    # sd_constraints = sd_miner.mine()

    # 挖掘否定依赖约束
    dd_miner = DenialDependencyMiner(df, strategy)
    dd_miner.generate_predicates()
    dd_miner.generate_candidates()
    dd_constraints = dd_miner.validate_all(min_support, min_conf, sort_by="confidence")


    # td_miner = TrendDependencyMiner(df, window_size=window, confidence_threshold=confidence_threshold)
    # td_constraints = td_miner.mine()



    return {
        "row_constraints": row_constraints,
        "speed_constraints": speed_constraints,
        "accel_constraints": accel_constraints,
        "variance_constraints": variance_constraints,
        # "amplitude_constraints": amplitude_constraints,
        "od_constraints": od_constraints,
        # "sd_constraints": sd_constraints,
        "dd_constraints": dd_constraints,
        # "td_constraints": td_constraints,
    }


def constraint_report(constraints):
    print("\n行约束:")
    if constraints["row_constraints"]:
        for c in constraints["row_constraints"]:
            print(c["formula"], "| loss:", round(c["loss_mse"], 3), "| support:", round(c["support"], 3))

    # 打印列约束
    print("\n速度约束:")
    if constraints["speed_constraints"]:
        for col, (min_val, max_val) in constraints["speed_constraints"].items():
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")

    print("\n加速度约束:")
    if constraints["accel_constraints"]:
        for col, (min_val, max_val) in constraints["accel_constraints"].items():
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")

    print("\n方差约束:")
    if constraints["variance_constraints"]:
        for col, (min_val, max_val) in constraints["variance_constraints"].items():
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")

    # print("\n振幅约束:")
    # if constraints["amplitude_constraints"]:
    #     for col, (min_val, max_val) in constraints["amplitude_constraints"].items():
    #         print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")

    print("\n=== 单调约束 ===")
    if constraints["od_constraints"]:
        for constraint in constraints["od_constraints"]:
            print(f"  {constraint['source']} → {constraint['target']}")
            print(f"    相关性: {constraint['correlation']:.3f}")
            print(f"    单调性: {constraint['monotonicity']}")
            print()

    # print("\n=== 顺序约束 ===")
    # if constraints["sd_constraints"]:
    #     for constraint in constraints["sd_constraints"]:
    #         print(f"  {constraint['M']} => {constraint['N']}")
    #         print(f"    置信度: {constraint['confidence']:.3f}")
    #         print(f"    M方向: {constraint['M_direction']}")
    #         print(f"    N方向: {constraint['N_direction']}")
    #         print(f"    范围: [{constraint['segment'][0]:.3f}, {constraint['segment'][1]:.3f}]")
    #         print(f"    区间长度: {constraint['length']}")
    #         print()

    print("\n=== 否定约束 ===")
    if constraints["dd_constraints"]:
        for constraint in constraints["dd_constraints"]:
            print(
                constraint["dd"],
                "support:",
                f"{constraint['support']:.3f}",
                "confidence:",
                f"{constraint['confidence']:.3f}",
            )

    # print("\n=== 趋势约束 ===")
    # if constraints["td_constraints"]:
    #     for constraint in constraints["td_constraints"]:
    #         print(f"  {constraint['M']} => {constraint['N']}")
    #         print(f"    置信度: {constraint['trend_confidence']:.3f}")
    #         print(f"    同步方向: {constraint['trend_type']}")

    #         print()


# 示例
if __name__ == "__main__":
    # 设置随机种子以便复现结果（可选）
    import random
    random.seed(42)

    # # 初始化列表
    # A = []
    # B = []
    # C = []
    # D = []

    # # 初始值
    # a_value = 10.0

    # for _ in range(100):
    #     # A: 时序数据，缓慢上升 + 微小波动
    #     a_value += random.uniform(0.5, 1.5)
    #     a_value += random.uniform(-0.3, 0.3)
    #     a_value = round(a_value, 2)
    #     A.append(a_value)

    #     # B = 2 * A + 噪声（±0.1以内）
    #     b_clean = 2 * A[-1]
    #     b_noisy = b_clean + random.uniform(-0.1, 0.1)
    #     B.append(round(b_noisy, 2))

    #     # C: 随机值 0~10
    #     c_value = round(random.uniform(0, 10), 2)
    #     C.append(c_value)

    #     # D = A + B + C + 噪声（±0.1以内）
    #     d_clean = A[-1] + B[-1] + C[-1]
    #     d_noisy = d_clean + random.uniform(-0.1, 0.1)
    #     D.append(round(d_noisy, 2))

    # # 构建数据字典
    # data = {
    #     "A": A,
    #     "B": B,
    #     "C": C,
    #     "D": D
    # }








    data = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    }
    df = pd.DataFrame(data)

    # np.random.seed(42)
    # df = pd.DataFrame(
    #     {
    #         "timestamp": range(100),
    #         "temperature": np.random.normal(20, 5, 100),
    #         "humidity": np.random.uniform(30, 70, 100),
    #     }
    # )

    # df = pd.read_csv("/home/yyy/TSC/TSClean/AutoClean/Datasets/IDF_Power/IDF_Power_Dirty.csv")

    attr_num = df.shape[1]
    constraints = mine_all_constraints(
        df,
        degree=3,
        attr_num=attr_num,
        window=20,
        strategy="auto",
        min_support=0.1,
        min_conf=0.95,
        confidence_threshold=0.7
    )

    # print("约束信息:",constraints)
    constraint_report(constraints)
