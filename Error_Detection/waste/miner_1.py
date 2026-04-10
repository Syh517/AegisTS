import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import itertools
from sklearn import linear_model
import difflib
import math
from itertools import combinations


class RowConstraintMiner:
    """
    行约束挖掘器（PolynomialFeatures + Ridge）
    constraints 直接包含公式字符串
    """

    def __init__(self, df, degree=2, alpha=1.0, tol=0.05, loss_threshold=0.01):
        """
        :param df: 待挖掘的数据
        :param degree: 多项式特征的最高阶
        :param alpha: Ridge 回归正则化系数
        :param tol: 约束容差
        :param loss_threshold: 约束损失阈值
        """
        self.df = df
        self.degree = degree
        self.alpha = alpha
        self.tol = tol
        self.loss_threshold = loss_threshold
        self.constraints = []  # 直接存约束字符串

    def row_miner(self, attr_num=2):
        """
        挖掘行约束
        :param attr_num: 每条约束参与回归的变量数量（包括目标变量）
        """
        df = self.df.copy()
        features = df.columns
        constraints = []

        # 遍历每种 attr_num 组合
        for combo in combinations(features, attr_num):
            for target in combo:
                # 参与回归的自变量
                X_cols = [c for c in combo if c != target]
                if not X_cols:
                    continue  # 至少需要一个自变量

                X = df[X_cols].values
                y = df[target].values

                # 多项式扩展
                poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                X_poly = poly.fit_transform(X)
                feature_names = poly.get_feature_names_out(X_cols)

                # Ridge 拟合
                model = Ridge(alpha=self.alpha, fit_intercept=True)
                model.fit(X_poly, y)

                # 预测误差
                y_pred = model.predict(X_poly)
                error = y - y_pred
                lower, upper = np.min(error) - self.tol, np.max(error) + self.tol
                loss = mean_squared_error(y, y_pred)

                # 构造公式字符串
                terms = []
                linear_terms = []
                nonlinear_terms = []

                coef_dict = {}  # 保存每个特征的系数
                for coef, name in zip(model.coef_, feature_names):
                    if abs(coef) < 1e-6:
                        continue
                    coef_dict[name] = coef
                    if "^" in name or "*" in name:
                        nonlinear_terms.append(f"{coef:.3f}*{name}")
                    else:
                        linear_terms.append(f"{coef:.3f}*{name}")

                intercept = model.intercept_
                if abs(intercept) > 1e-6:
                    terms.append(f"{intercept:.3f}")

                terms = linear_terms + nonlinear_terms + terms
                constraint_str = (
                    f"{lower:.3f} <= {target} - ({' + '.join(terms)}) <= {upper:.3f}"
                )

                # 保存完整信息到 constraints
                constraints.append(
                    {
                        "formula": constraint_str,  # 公式字符串
                        "target": target,  # 目标变量
                        "features": X_cols,  # 特征变量列表
                        "coef": coef_dict,  # 系数字典
                        "intercept": intercept,  # 截距
                        "lower": lower,  # 下界
                        "upper": upper,  # 上界
                        "loss": loss,  # 均方误差
                    }
                )

        constraints_sorted = sorted(constraints, key=lambda x: x["loss"])
        for c in constraints_sorted:
            # 舍弃高 loss
            if c["loss"] > self.loss_threshold:
                continue

            # 保留约束
            self.constraints.append(c)

        return self.constraints


class ColConstraintMiner:
    """列约束挖掘器"""

    def __init__(self, df, q_low=0.01, q_high=0.99):
        self.df = df
        self.q_low = q_low
        self.q_high = q_high

    def _calc_speed(self, series):
        diffs = series.diff(periods=1).fillna(0)
        return (diffs.quantile(self.q_low), diffs.quantile(self.q_high))
    
    def _calc_accel(self, series):
        diffs = series.diff(periods=1).diff(periods=1).fillna(0)
        return (diffs.quantile(self.q_low), diffs.quantile(self.q_high))

    def _calc_amplitude(self, series):
        return (series.quantile(self.q_low), series.quantile(self.q_high))

    def _calc_variance(self, series, tolerance=0.5):
        var_val = np.var(series.dropna(), ddof=1)
        return (var_val * (1 - tolerance), var_val * (1 + tolerance))

    def mine_col_constraints(self, time_col="timestamp"):
        speed_constraints = {}
        accel_constraints = {}
        variance_constraints = {}
        amplitude_constraints = {}
        for col in self.df.columns:
            if col != time_col:
                speed_constraints[col] = self._calc_speed(self.df[col])
                accel_constraints[col] = self._calc_accel(self.df[col])
                variance_constraints[col] = self._calc_variance(self.df[col])
                amplitude_constraints[col] = self._calc_amplitude(self.df[col])
        return (
            speed_constraints,
            accel_constraints,
            variance_constraints,
            amplitude_constraints,
        )


class OrderDependencyMiner:
    """
    Order Dependency (OD) 约束挖掘器
    
    定义：如果属性 A 的取值排序决定了属性 B 的排序（单调关系），则 A → B 是一个 OD。
    例子：在时序数据中，如果时间戳递增，那么"库存量"也单调不减。
    """

    def __init__(self, df, method='kendall', threshold=0.9):
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
        if self.method == 'kendall':
            correlation = series_a.corr(series_b, method='kendall')
        elif self.method == 'spearman':
            correlation = series_a.corr(series_b, method='spearman')
        else:
            raise ValueError("Method must be 'kendall' or 'spearman'")
            
        is_monotonic = abs(correlation) >= self.threshold
        return is_monotonic, correlation

    def _check_order_dependency(self, series_a, series_b):
        """
        排序检测（验证是否递增/递减单调）
        """
        clean_data = pd.DataFrame({'A': series_a, 'B': series_b}).dropna()
        if len(clean_data) < 2:
            return False, 'none', 0, 0

        # 按 A 排序，检查 B 的单调性
        sorted_data = clean_data.sort_values('A')
        b_values = sorted_data['B'].values

        # 递增与递减违例
        inc_violations = np.sum(np.diff(b_values) < 0)
        dec_violations = np.sum(np.diff(b_values) > 0)
        total_pairs = len(b_values) - 1

        inc_ratio = inc_violations / total_pairs
        dec_ratio = dec_violations / total_pairs

        # 判断递增还是递减
        if inc_ratio <= (1 - self.threshold):
            return True, 'increasing', inc_ratio, inc_violations
        elif dec_ratio <= (1 - self.threshold):
            return True, 'decreasing', dec_ratio, dec_violations
        else:
            return False, 'none', min(inc_ratio, dec_ratio), min(inc_violations, dec_violations)

    def mine_order_dependencies(self):
        """
        挖掘所有属性之间的 OD 关系（支持递增/递减）
        """
        columns = self.df.columns
        od_constraints = []

        for col_a in columns:
            for col_b in columns:
                if col_a == col_b:
                    continue

                # 方法1: 相关性筛选
                is_monotonic, correlation = self._check_monotonicity(
                    self.df[col_a], self.df[col_b]
                )

                # 方法2: 排序检测（只要两者之一成立，就保留约束）
                is_dep, monotonicity, violation_ratio, violations = self._check_order_dependency(
                    self.df[col_a], self.df[col_b]
                )

                if is_monotonic or is_dep:
                    constraint = {
                        'source': col_a,
                        'target': col_b,
                        'correlation': correlation,
                        'violations': violations,
                        'violation_ratio': violation_ratio,
                        'monotonicity': monotonicity if monotonicity != 'none' else (
                            'increasing' if correlation > 0 else 'decreasing'
                        )
                    }
                    od_constraints.append(constraint)

        self.od_constraints = od_constraints
        return od_constraints


class SequentialDependencyMiner:
    def __init__(self, df, confidence_threshold=0.95, window=None):
        self.df = df.copy()
        self.confidence_threshold = confidence_threshold
        self.window = window  # 窗口大小（行数）
        self.results = []

    def _mine_sd(self, M, N, df_segment):
        df_sorted = df_segment.sort_values(by=M).reset_index(drop=True)
        delta = df_sorted[N].diff().iloc[1:]
        g = (delta.min(), delta.max())
        valid_idx = delta.between(g[0], g[1])
        confidence = valid_idx.mean()

        if confidence >= self.confidence_threshold:
            ranges = []
            start_idx = None
            for i, is_valid in enumerate(valid_idx):
                if is_valid and start_idx is None:
                    start_idx = i
                elif not is_valid and start_idx is not None:
                    ranges.append((df_sorted[M[0]].iloc[start_idx], df_sorted[M[0]].iloc[i]))
                    start_idx = None
            if start_idx is not None:
                ranges.append((df_sorted[M[0]].iloc[start_idx], df_sorted[M[0]].iloc[len(valid_idx)]))
            return {'M': M, 'N': N, 'g': g, 'confidence': confidence, 'ranges': ranges}
        return None

    def mine(self):
        numeric_cols = self.df.select_dtypes(include=[np.number, 'datetime']).columns.tolist()
        candidate_M = [col for col in numeric_cols if np.issubdtype(self.df[col].dtype, np.datetime64) or
                        np.all(np.diff(self.df[col].values) >= 0)]
        candidate_N = [col for col in numeric_cols if col not in candidate_M]

        n_rows = len(self.df)
        step = self.window if self.window is not None else n_rows

        for start in range(0, n_rows, step):
            end = min(start + step, n_rows)
            df_window = self.df.iloc[start:end]

            for M in candidate_M:
                for N in candidate_N:
                    result = self._mine_sd([M], N, df_window)
                    # print(result)
                    if result:
                        self.results.append(result)
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

    def get_n_bins(self, col):
        """根据策略确定某个变量的分箱数"""
        if isinstance(self.n_bins, dict):
            return self.n_bins.get(col, 3)
        elif isinstance(self.n_bins, int):
            return self.n_bins
        elif self.strategy == "auto":
            N = len(self.df[col])
            iqr = np.percentile(self.df[col], 75) - np.percentile(self.df[col], 25)
            bin_width = 2 * iqr / (N ** (1/3) + 1e-6)
            n_bins = max(2, int((self.df[col].max() - self.df[col].min()) / (bin_width + 1e-6)))
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
            p1, p2 = inner[:split_idx], inner[split_idx + 5:]

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

            results.append({
                "dd": c,
                "support": support,
                "confidence": conf
            })

        # 排序
        if sort_by == "confidence":
            results.sort(key=lambda x: (-x["confidence"], -x["support"]))
        elif sort_by == "support":
            results.sort(key=lambda x: (-x["support"], -x["confidence"]))

        return results






def mine_all_constraints(df, degree=3, attr_num=3, outlier_rate=0.1, confidence_threshold=0.95, window=10, n_bins=3, strategy="auto", min_support=0.05, min_conf=0.9):
    row_miner = RowConstraintMiner(df, degree)
    row_constraints = row_miner.row_miner(attr_num)

    q_low = outlier_rate/4 
    q_high = 1 - outlier_rate/4
    col_miner = ColConstraintMiner(df, q_low=q_low, q_high=q_high)
    (
        speed_constraints,
        accel_constraints,
        variance_constraints,
        amplitude_constraints,
    ) = col_miner.mine_col_constraints()



    od_miner = OrderDependencyMiner(df)
    od_constraints = od_miner.mine_order_dependencies()


    # 挖掘顺序依赖约束
    sd_miner = SequentialDependencyMiner(df, confidence_threshold, window)
    sd_constraints = sd_miner.mine()
    
    # 挖掘否定依赖约束
    dd_miner = DenialDependencyMiner(df, strategy)
    dd_miner.generate_predicates()
    dd_miner.generate_candidates()
    dd_constraints = dd_miner.validate_all(min_support, min_conf, sort_by="confidence")

    return {
        "row_constraints": row_constraints,
        "speed_constraints": speed_constraints,
        "accel_constraints": accel_constraints,
        "variance_constraints": variance_constraints,
        "amplitude_constraints": amplitude_constraints,
        "od_constraints": od_constraints,
        "sd_constraints": sd_constraints,
        "dd_constraints": dd_constraints,
    }





def constraint_report(constraints):
    print("\n行约束:")
    if constraints["row_constraints"]:
        for c in constraints["row_constraints"]:
            print(c["formula"], "| loss:", round(c["loss"], 3))

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

    print("\n振幅约束:")
    if constraints["amplitude_constraints"]:
        for col, (min_val, max_val) in constraints["amplitude_constraints"].items():
            print(f"{col}: [{min_val:.3f}, {max_val:.3f}]")

    print("\n=== 单调约束 ===")
    if constraints["od_constraints"]:
        for constraint in constraints['od_constraints']:
            print(f"  {constraint['source']} → {constraint['target']}")
            print(f"    相关性: {constraint['correlation']:.3f}")
            print(f"    单调性: {constraint['monotonicity']}")
            # print(f"    方法: {constraint['method']}")
            print()

    print("\n=== 顺序约束 ===")
    if constraints["sd_constraints"]:
        for constraint in constraints["sd_constraints"]:
            print(f"  {constraint['M']} => {constraint['N']}")
            print(f"    置信度: {constraint['confidence']:.3f}")
            print(f"    范围: [{constraint['g'][0]:.3f}, {constraint['g'][1]:.3f}]")
            print(f"    有效区间: {constraint['ranges']}")
            print()


    print("\n=== 否定约束 ===")
    if constraints["dd_constraints"]:
        for constraint in constraints["dd_constraints"]:
                print(constraint["dd"], "support:", f"{constraint['support']:.3f}", "confidence:", f"{constraint['confidence']:.3f}")



# 示例
if __name__ == "__main__":
    # data = {
    #     "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    #     "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
    #     "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    # }
    # df = pd.DataFrame(data)


    np.random.seed(42)
    df = pd.DataFrame({
        "timestamp": range(100),
        "temperature": np.random.normal(20, 5, 100),
        "humidity": np.random.uniform(30, 70, 100)
    })

    constraints = mine_all_constraints(df, degree=3, attr_num=2, window=20, strategy="auto", min_support=0.1, min_conf=0.9)

    constraint_report(constraints)