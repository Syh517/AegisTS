import pandas as pd
import numpy as np
from itertools import combinations

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
                    print(result)
                    if result:
                        self.results.append(result)
        return self.results


class DenialDependencyMiner:
    def __init__(self, df, n_bins=3, strategy="fixed"):
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

    def validate_all(self, min_support=0.1, min_conf=0.9,  sort_by="confidence"):
        """
        验证并筛选DD
        :param min_support: 最小覆盖率
        :param min_conf: 最小置信度
        :param window_size: 滑动窗口大小 (None 表示不用)
        :param window_consistency: 窗口中需要成立的比例
        :param sort_by: "confidence" 或 "support"
        """
        results = []
        N = len(self.df)

        for c in self.candidates:
            p1, p2 = c.strip("¬()").split(" and ", 1)
            subset = self.df.query(f"{p1} and {p2}")

            support = len(subset) / N
            violations = len(subset)
            conf = 1 - violations / N

            # print(f"验证DD: {c}", f"support: {support:.3f}", f"confidence: {conf:.3f}")

            # 基本筛选
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



def mine_all_constraints(df, confidence_threshold=0.95, window=10):
    """
    挖掘所有约束条件，包括顺序依赖和否定依赖
    """
    # 挖掘顺序依赖约束
    sd_miner = SequentialDependencyMiner(df, confidence_threshold, window)
    sd_constraints = sd_miner.mine()
    

    return {
        "sd_constraints": sd_constraints,
    }


def constraint_report(constraints):
    """
    打印约束报告，格式与miner_1.py保持一致
    """
    print("\n=== 顺序约束 ===")
    if constraints["sd_constraints"]:
        for constraint in constraints["sd_constraints"]:
            print(f"  {constraint['M']} => {constraint['N']}")
            print(f"    置信度: {constraint['confidence']:.3f}")
            print(f"    范围: [{constraint['g'][0]:.3f}, {constraint['g'][1]:.3f}]")
            print(f"    有效区间: {constraint['ranges']}")
            print()
    else:
        print("  未发现顺序依赖约束")



if __name__ == "__main__":
    data = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    }
    df = pd.DataFrame(data)

    constraints = mine_all_constraints(df, confidence_threshold=0.95, window=10)

    constraint_report(constraints)