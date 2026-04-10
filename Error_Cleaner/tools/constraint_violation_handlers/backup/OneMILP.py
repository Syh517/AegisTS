import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import gurobipy as gp
from gurobipy import GRB
import warnings


class OneMILP:
    def __init__(self, speed_constraints: Dict[str, Tuple[float, float]], T: int = 5):
        """
        :param speed_constraints: {col: (SMIN, SMAX)}
        :param T: 默认时间窗口
        """
        self.speed_constraints = speed_constraints
        self.T = T
        self.M = 1000.0  # Big-M 常数

    def mainGlobal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('timestamp').reset_index(drop=True).copy()
        n = len(df)

        if n < 2:
            for col in self.speed_constraints:
                df[f"{col}_repaired"] = df[col]
            df["is_fixed"] = False
            return df

        timestamps = df["timestamp"].values.astype(int)
        all_cols = list(self.speed_constraints.keys())
        org_vals = {col: df[col].values.astype(float) for col in all_cols}

        repaired_vals, is_fixed_array = self._build_joint_milp(org_vals, timestamps, all_cols)

        df_repaired = df.copy()
        for col in all_cols:
            df_repaired[f"{col}_repaired"] = repaired_vals[col]
        df_repaired["is_fixed"] = is_fixed_array

        return df_repaired

    def _build_joint_milp(self, org_vals, timestamps, columns):
        n = len(timestamps)
        model = gp.Model("Multi_OneMILP")
        model.setParam("OutputFlag", 0)

        # 修复变量
        x_vars = {
            col: model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY,
                               vtype=GRB.CONTINUOUS, name=f"x_{col}")
            for col in columns
        }
        # 时间点是否被修改
        z_vars = model.addVars(n, vtype=GRB.BINARY, name="z")

        # 目标：最少修复的时间点数量
        model.setObjective(gp.quicksum(z_vars[i] for i in range(n)), GRB.MINIMIZE)

        # ——添加速度约束——
        for col in columns:
            SMIN, SMAX = self.speed_constraints[col]

            for i in range(n):
                ti = timestamps[i]
                for j in range(i + 1, n):
                    tj = timestamps[j]
                    dt = tj - ti
                    if dt > self.T:  # 默认窗口
                        break

                    model.addConstr(
                        x_vars[col][j] - x_vars[col][i] <= SMAX * dt,
                        name=f"speed_up_{col}_{i}_{j}"
                    )
                    model.addConstr(
                        x_vars[col][j] - x_vars[col][i] >= SMIN * dt,
                        name=f"speed_low_{col}_{i}_{j}"
                    )

        # ——Big-M 链接原值与修复值——
        for col in columns:
            vals = org_vals[col]
            for i in range(n):
                model.addConstr(
                    x_vars[col][i] <= vals[i] + self.M * z_vars[i],
                    name=f"bigM_ub_{col}_{i}"
                )
                model.addConstr(
                    x_vars[col][i] >= vals[i] - self.M * z_vars[i],
                    name=f"bigM_lb_{col}_{i}"
                )

        # 求解
        try:
            model.optimize()
        except gp.GurobiError as e:
            print("Gurobi error:", e)
            raise

        # 未找到可行解
        if model.status != GRB.OPTIMAL:
            warnings.warn(f"Multi-OneMILP failed! status={model.status}")
            return org_vals, np.zeros(n, dtype=bool)

        # 提取修复结果
        repaired = {
            col: np.array([x_vars[col][i].X for i in range(n)])
            for col in columns
        }
        is_fixed = np.array([z_vars[i].X > 0.5 for i in range(n)])

        # 对非常接近的值贴回原始
        for col in columns:
            orig = org_vals[col]
            fix = repaired[col]
            repaired[col][np.abs(fix - orig) < 1e-3] = orig[np.abs(fix - orig) < 1e-3]

        print(f"Multi-OneMILP Obj: {model.objVal:.6f} time points modified")

        return repaired, is_fixed


if __name__ =="__main__":
    df = pd.DataFrame({
        "timestamp": [0,1,2,3,4,5,6],
        "lon": [0,0.5,1.0,5.5,6.0,6.5,7.0],   # 中间有个跳变点(→异常)
        "lat": [0,0.3,0.6,0.9,1.0,1.1,1.2],
        "speed": [5,6,7,100,6,5,5]
    })

    speed_constraints = {
        "lon": (-0.8, 0.8),
        "lat": (-0.6, 0.6),
        "speed": (0.0, 20.0)
    }

    solver = OneMILP(speed_constraints)
    res = solver.mainGlobal(df)
    print(res)