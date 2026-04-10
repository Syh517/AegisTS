from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional


class BaseCleaningAlgorithm(ABC):
    @abstractmethod
    def clean(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class MTSClean(BaseCleaningAlgorithm):
    """
    多变量时序数据清洗器，支持：
      - 行约束（线性不等式）
      - 速度约束（一阶差分）
      - 加速度约束（二阶差分）
    """

    def __init__(self):
        pass

    def _align_coefficients(self, coef_dict: Dict[str, float], columns: List[str]) -> np.ndarray:
        return np.array([coef_dict.get(col, 0.0) for col in columns], dtype=float)

    def _check_row_violations(self, row: np.ndarray, constraints: List[Tuple[np.ndarray, float, float]]) -> bool:
        for coefs, rho_min, rho_max in constraints:
            val = np.dot(coefs, row)
            if val < rho_min or val > rho_max:
                return True
        return False

    def clean(
        self,
        data: pd.DataFrame,
        *,
        row_constraints: Optional[List[Tuple[str, Dict[str, float], float, float]]] = None,
        speed_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        acceleration_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        row_cleaning_rounds: int = 4,
        **kwargs
    ) -> pd.DataFrame:
        """
        清洗主入口。

        Parameters
        ----------
        data : pd.DataFrame
            输入时序数据（不含时间戳列）
        row_constraints : list of (desc, coef_dict, min, max)
        speed_constraints : dict {col: (v_min, v_max)}
        acceleration_constraints : dict {col: (a_min, a_max)}
        row_cleaning_rounds : int, default=4
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input 'data' must be a non-empty pandas DataFrame.")

        # 默认空约束
        row_constraints = row_constraints or []
        speed_constraints = speed_constraints or {}
        acceleration_constraints = acceleration_constraints or {}

        # 验证列存在性
        all_cols = set(data.columns)
        for d in [speed_constraints, acceleration_constraints]:
            if not set(d.keys()).issubset(all_cols):
                raise ValueError("Constraint contains unknown column(s).")

        # 对齐行约束系数
        aligned_row_cons = []
        for _, coef_dict, rmin, rmax in row_constraints:
            coefs = self._align_coefficients(coef_dict, data.columns.tolist())
            aligned_row_cons.append((coefs, rmin, rmax))

        n_rows, n_cols = data.shape
        total_steps = (
            row_cleaning_rounds * n_rows
            + (n_rows - 1) * len(speed_constraints)
            + (n_rows - 2) * len(acceleration_constraints)
        )

        with tqdm(total=total_steps, desc="MTSClean") as pbar:
            cleaned = data.copy()

            # Step 1: 行约束清洗（迭代）
            for _ in range(row_cleaning_rounds):
                cleaned = self._clean_with_row_constraints(cleaned, aligned_row_cons, pbar)

            # Step 2: 速度约束清洗（因果，使用已清洗数据）
            if speed_constraints:
                cleaned = self._clean_with_speed_constraints(cleaned, speed_constraints, pbar)

            # Step 3: 加速度约束清洗（因果，使用已清洗数据）
            if acceleration_constraints:
                cleaned = self._clean_with_acceleration_constraints(cleaned, acceleration_constraints, pbar)

        return cleaned

    def _clean_with_row_constraints(
        self,
        data: pd.DataFrame,
        constraints: List[Tuple[np.ndarray, float, float]],
        pbar: tqdm
    ) -> pd.DataFrame:

        n_rows, n_cols = data.shape
        result = np.empty_like(data.values, dtype=float)

        for i in range(n_rows):
            row = data.iloc[i].values

            # 如果不违反约束，直接跳过
            if not self._check_row_violations(row, constraints):
                result[i] = row
                pbar.update(1)
                continue

            c = np.ones(2 * n_cols)
            A_ub, b_ub = [], []

            for coefs, rmin, rmax in constraints:
                dot_val = np.dot(coefs, row)

                # -------- 处理 max 约束 coefs·x <= rmax ----------
                if not np.isinf(rmax):
                    # coefs·(row + u - v) <= rmax  →  coefs·u - coefs·v <= rmax - dot
                    ext = np.hstack([coefs, -coefs])
                    A_ub.append(ext)
                    b_ub.append(rmax - dot_val)

                # -------- 处理 min 约束 coefs·x >= rmin ----------
                if not np.isinf(rmin):
                    # coefs·x >= rmin  →  -coefs·x <= -rmin
                    ext = np.hstack([-coefs, coefs])
                    A_ub.append(ext)
                    b_ub.append(dot_val - rmin)

            bounds = [(0, None)] * (2 * n_cols)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if res.success:
                u = res.x[:n_cols]
                v = res.x[n_cols:]
                result[i] = row + u - v
            else:
                result[i] = row  # LP失败则回退

            pbar.update(1)

        return pd.DataFrame(result, columns=data.columns, index=data.index)


    def _clean_with_speed_constraints(
        self,
        data: pd.DataFrame,
        speed_bounds: Dict[str, Tuple[float, float]],
        pbar: tqdm
    ) -> pd.DataFrame:
        cleaned = data.copy()
        for col in speed_bounds:
            v_min, v_max = speed_bounds[col]
            series = cleaned[col].values.copy()
            for i in range(1, len(series)):
                pred_low = series[i - 1] + v_min
                pred_high = series[i - 1] + v_max
                series[i] = np.clip(series[i], pred_low, pred_high)
                pbar.update(1)
            cleaned[col] = series
        return cleaned

    def _clean_with_acceleration_constraints(
        self,
        data: pd.DataFrame,
        acc_bounds: Dict[str, Tuple[float, float]],
        pbar: tqdm
    ) -> pd.DataFrame:
        """
        加速度约束：x[t] - 2*x[t-1] + x[t-2] ∈ [a_min, a_max]
        => x[t] ∈ [2*x[t-1] - x[t-2] + a_min, 2*x[t-1] - x[t-2] + a_max]
        """
        cleaned = data.copy()
        for col in acc_bounds:
            a_min, a_max = acc_bounds[col]
            series = cleaned[col].values.copy()
            for t in range(2, len(series)):
                pred_center = 2 * series[t - 1] - series[t - 2]
                low = pred_center + a_min
                high = pred_center + a_max
                series[t] = np.clip(series[t], low, high)
                pbar.update(1)
            cleaned[col] = series
        return cleaned


# ==============================
# 测试用例（增强：包含加速度异常）
# ==============================

def test_mtsclean_full():
    print("=" * 60)
    print("MTSClean 完整测试：行约束 + 速度约束 + 加速度约束")
    print("=" * 60)

    np.random.seed(42)
    n = 100
    t = np.arange(n)
    temp = 20 + 10 * np.sin(0.2 * t) + np.random.normal(0, 0.3, n)
    press = 100 + 0.3 * t + np.random.normal(0, 0.8, n)

    df = pd.DataFrame({
        'timestamp': t,
        'temperature': temp,
        'pressure': press
    })

    # 注入三类异常
    df.loc[30, 'temperature'] += 15          # 速度突变
    df.loc[50, ['temperature', 'pressure']] = [45, 210]  # 行约束违反
    df.loc[70, 'temperature'] += 10          # 制造加速度异常（70比69高很多，但71正常）

    print("原始数据异常注入完成。")

    # 约束定义（使用字典，列安全）
    row_cons = [
        ("0.5*T + 0.3*P <= 80", {'temperature': 0.5, 'pressure': 0.3}, -np.inf, 80),
        ("T + P <= 200", {'temperature': 1.0, 'pressure': 1.0}, -np.inf, 200),
        ("50 <= 0.8*T + 0.2*P <= 120", {'temperature': 0.8, 'pressure': 0.2}, 50,  120)
    ]

    speed_cons = {'temperature': (-2.0, 2.0), 'pressure': (-3.0, 3.0)}
    acc_cons = {'temperature': (-1.0, 1.0)}  # 温度加速度不能突变太剧烈

    cleaner = MTSClean()
    cleaned = cleaner.clean(
        df[['temperature', 'pressure']],
        row_constraints=row_cons,
        speed_constraints=speed_cons,
        acceleration_constraints=acc_cons,
        row_cleaning_rounds=3
    )

    # 合并结果
    out = pd.DataFrame({
        't': df['timestamp'],
        'T_orig': df['temperature'],
        'T_clean': cleaned['temperature'],
        'P_orig': df['pressure'],
        'P_clean': cleaned['pressure']
    })

    # 打印关键点
    print(f"\n第30行（速度异常）: {out.loc[30, 'T_orig']:.2f} → {out.loc[30, 'T_clean']:.2f}")
    print(f"第50行（行约束）  : {0.5*out.loc[50,'T_orig']+0.3*out.loc[50,'P_orig']:.2f} → "
          f"{0.5*out.loc[50,'T_clean']+0.3*out.loc[50,'P_clean']:.2f}")
    print(f"第70行（加速度）  : {out.loc[70, 'T_orig']:.2f} → {out.loc[70, 'T_clean']:.2f}")

    # 验证约束
    T, P = cleaned['temperature'], cleaned['pressure']
    print("\n清洗后违反次数:")
    print("行约束1:", ((0.5*T + 0.3*P) > 80).sum())
    print("速度约束(T):", ((T.diff().fillna(0).abs() > 2.0)).sum())
    print("加速度约束(T):", ((T.diff().diff().fillna(0).abs() > 1.0)).sum())

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    test_mtsclean_full()