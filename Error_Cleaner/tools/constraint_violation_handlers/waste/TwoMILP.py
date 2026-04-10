import pandas as pd
import numpy as np
from gurobipy import Model, GRB

class TwoMILP:
    """
    双列联合轨迹修复器，支持 speed_constraints 字典格式：
        speed_constraints = {
            'lon': (-0.8, 0.8),
            'lat': (-0.6, 0.6)
        }
    默认时间窗口 T=5
    """

    def __init__(self, df: pd.DataFrame, time_col: str, speed_constraints: dict, T: int = 5, M: float = None):
        self.df = df.sort_values(by=time_col).reset_index(drop=True)
        self.time_col = time_col
        self.speed_constraints = speed_constraints
        self.T = T
        self.cols = list(speed_constraints.keys())
        self.n = len(df)

        # 原始值
        self.org_vals = {col: df[col].values.astype(float) for col in self.cols}
        self.times = df[time_col].values.astype(int)

        # 自动估算 M
        if M is None:
            ranges = [self.org_vals[col].max() - self.org_vals[col].min() for col in self.cols]
            self.M = 10 * max(ranges)
        else:
            self.M = M

        self.model = None
        self.repaired = {col: self.org_vals[col].copy() for col in self.cols}
        self.is_fixed = np.zeros(self.n, dtype=bool)

    def build(self):
        model = Model("TwoMILP")
        model.setParam(GRB.Param.OutputFlag, 0)

        # 创建变量
        x_vars = {col: model.addVars(self.n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=col)
                  for col in self.cols}
        z_vars = model.addVars(self.n, vtype=GRB.BINARY, name="z")

        # 目标：最小化修改点数量
        model.setObjective(z_vars.sum(), GRB.MINIMIZE)

        # 速度约束
        for col in self.cols:
            SMIN, SMAX = self.speed_constraints[col]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    dt = self.times[j] - self.times[i]
                    if dt > self.T:
                        break
                    model.addConstr(x_vars[col][j] - x_vars[col][i] <= SMAX * dt, name=f"speed_up_{col}_{i}_{j}")
                    model.addConstr(x_vars[col][j] - x_vars[col][i] >= SMIN * dt, name=f"speed_low_{col}_{i}_{j}")

        # Big-M 约束
        for col in self.cols:
            vals = self.org_vals[col]
            for i in range(self.n):
                model.addConstr(x_vars[col][i] <= vals[i] + self.M * z_vars[i], name=f"bigM_ub_{col}_{i}")
                model.addConstr(x_vars[col][i] >= vals[i] - self.M * z_vars[i], name=f"bigM_lb_{col}_{i}")

        # 求解
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"Warning: Model not optimal. Status={model.status}")
        else:
            self.repaired = {col: np.array([x_vars[col][i].X for i in range(self.n)]) for col in self.cols}
            self.is_fixed = np.array([z_vars[i].X > 0.5 for i in range(self.n)])

        self.model = model
        return self

    def get_result(self, tol=1e-3):
        df_out = self.df.copy()
        for col in self.cols:
            close_mask = np.abs(self.repaired[col] - self.org_vals[col]) < tol
            self.repaired[col][close_mask] = self.org_vals[col][close_mask]
            df_out[f"{col}_repaired"] = self.repaired[col]

        df_out["is_fixed"] = self.is_fixed.astype(int)
        print(f"Optimal objective: {self.model.ObjVal if self.model else 'N/A'}")
        print(f"Number of modified points: {self.is_fixed.sum()}")
        return df_out



if __name__ == "__main__":
    df = pd.DataFrame({
        "timestamp": [0,1,2,3,4,5],
        "lon": [0, 0.5, 1.0, 5.0, 5.5, 6.0],
        "lat": [0, 0.2, 0.4, 1.0, 1.1, 1.2]
    })

    speed_constraints = {
        "lon": (-0.8, 0.8),
        "lat": (-0.6, 0.6)
    }

    solver = TwoMILP(df, time_col="timestamp", speed_constraints=speed_constraints)
    solver.build()
    res = solver.get_result()
    print(res)
