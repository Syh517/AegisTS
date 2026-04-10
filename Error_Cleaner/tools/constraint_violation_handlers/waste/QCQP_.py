import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB
import warnings
warnings.filterwarnings("ignore")

class QCQP:
    """
    Python port of Java QCQP algorithm using Gurobi for trajectory cleaning.
    Ensures all inter-point distances satisfy: d <= S * |tj - ti| for |tj - ti| <= T.
    Only repairs spatial (x,y) columns; preserves all other columns.
    """
    
    def __init__(self, S: float, T: int, space_columns: List[str] = ['x', 'y']):
        """
        :param S: Maximum allowed speed (distance unit per time unit)
        :param T: Maximum time window (in time units) to enforce speed constraint
        :param space_columns: Names of spatial coordinate columns, e.g., ['x', 'y']
        """
        self.S = S
        self.T = T
        self.space_columns = space_columns

    def mainGlobal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: repairs the trajectory by solving a QCQP optimization.
        
        :param df: Input DataFrame with 'timestamp' and spatial columns (e.g., 'x', 'y')
        :return: Repaired DataFrame with new columns: x_repaired, y_repaired, is_fixed
        """
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True).copy()
        len_df = len(df)
        
        if len_df < 2:
            for col in self.space_columns:
                df[f"{col}_repaired"] = df[col]
            df['is_fixed'] = False
            return df

        # Extract original values and timestamps
        org_vals = df[self.space_columns].values.astype(float)  # shape: (n, 2)
        timestamps = df['timestamp'].values.astype(int)         # shape: (n,)

        # Build and solve QCQP model
        repaired_vals = self._build_qcqp(org_vals, timestamps)

        # Apply repaired values back
        df_repaired = df.copy()
        is_fixed = np.zeros(len_df, dtype=bool)
        for i in range(len_df):
            for j, col in enumerate(self.space_columns):
                old_val = df.loc[i, col]
                new_val = repaired_vals[i, j]
                # Check if value changed significantly
                if abs(new_val - old_val) > 1e-3:
                    is_fixed[i] = True
                # Snap back if very close to original (per Java logic)
                if abs(new_val - old_val) < 1e-3:
                    repaired_vals[i, j] = old_val

        # Assign repaired values
        for j, col in enumerate(self.space_columns):
            df_repaired[f"{col}_repaired"] = repaired_vals[:, j]
        df_repaired['is_fixed'] = is_fixed

        return df_repaired

    def _build_qcqp(self, org_vals: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Builds and solves the QCQP model using Gurobi.
        
        Minimize: sum_i ( (xi - x0i)^2 + (yi - y0i)^2 )
        Subject to: (xj - xi)^2 + (yj - yi)^2 <= S^2 * (tj - ti)^2   for all i < j where tj - ti <= T
        """
        n = len(org_vals)
        model = gp.Model("QCQP_Trajectory_Cleaning")
        model.setParam('OutputFlag', 0)  # Silent mode
        # model.setParam('NonConvex', 2)  # Enable non-convex QCQP (if needed)

        # Create variables: x[i], y[i]
        x_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
        y_vars = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

        # Objective: minimize sum of squared deviations
        obj = gp.QuadExpr()
        for i in range(n):
            xi, yi = org_vals[i]
            obj += (x_vars[i] - xi) ** 2
            obj += (y_vars[i] - yi) ** 2
        model.setObjective(obj, GRB.MINIMIZE)

        # Add quadratic constraints: (xj - xi)^2 + (yj - yi)^2 <= (S * (tj - ti))^2
        # Only for pairs within time window T
        count = 0
        for i in range(n):
            ti = timestamps[i]
            for j in range(i + 1, n):
                tj = timestamps[j]
                delta_t = tj - ti
                if delta_t > self.T:
                    break  # Since timestamps are sorted
                max_dist_sq = (self.S * delta_t) ** 2
                # Constraint: (x[j] - x[i])^2 + (y[j] - y[i])^2 <= max_dist_sq
                expr = (x_vars[j] - x_vars[i]) ** 2 + (y_vars[j] - y_vars[i]) ** 2
                model.addQConstr(expr <= max_dist_sq, name=f"qc_{count}")
                count += 1

        # Optimize
        try:
            model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            raise

        if model.status != GRB.OPTIMAL:
            warnings.warn(f"QCQP optimization failed with status {model.status}")
            # Return original values if failed
            return org_vals.copy()

        # Extract results
        repaired = np.zeros_like(org_vals)
        for i in range(n):
            repaired[i, 0] = x_vars[i].X
            repaired[i, 1] = y_vars[i].X

        print(f"QCQP Obj: {model.objVal:.6f}")
        return repaired

if __name__ =="__main__":
    # ===========================
    # 1. 构造测试轨迹数据
    # ===========================
    timestamps = np.arange(0, 10)  # 10个时间点
    x = np.cos(np.linspace(0, 2*np.pi, 10))
    y = np.sin(np.linspace(0, 2*np.pi, 10))

    # 制造异常点：t=5 的位置突然跳到远处
    x[5] += 5
    y[5] += 5

    df = pd.DataFrame({
        "timestamp": timestamps,
        "x": x,
        "y": y
    })

    print("==== 原始数据 ====")
    print(df)

    # ===========================
    # 2. 调用 QCQP 修复
    # ===========================
    # 最大速度限制 S=1.0
    # 时间窗口 T=3
    qcqp = QCQP(S=1.0, T=3, space_columns=['x', 'y'])

    df_repaired = qcqp.mainGlobal(df)

    # ===========================
    # 3. 输出结果
    # ===========================
    print("\n==== 修复后数据 ====")
    print(df_repaired)

    print("\n==== 被修复的点 ====")
    print(df_repaired[df_repaired['is_fixed'] == True])