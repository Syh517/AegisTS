import numpy as np
import pandas as pd
from collections import deque


class Clean4MTS:
    """
    Time series cleaner enforcing speed constraints.
    No regression rule involved.
    """
    def __init__(self, dataframe: pd.DataFrame, speed_constraint: dict, temporal_attr: str,
                 relevant_attrs: list, max_window_size=500, print_repair_message=False):
        # Validate inputs
        assert temporal_attr in dataframe.columns, f"{temporal_attr} not in dataframe"
        assert all(col in dataframe.columns for col in relevant_attrs), "Some relevant attrs not in dataframe"
        self.temporal_attr = temporal_attr
        self.var_omega = relevant_attrs  # All attributes to be checked
        self.dataframe = dataframe[self.var_omega]
        self.timestamps = dataframe[temporal_attr].values

        # Speed constraints
        for attr in speed_constraint:
            assert attr in self.var_omega, f"Speed constraint for unknown attr: {attr}"
        self.sc = speed_constraint

        # Precompute bounds
        var_omega_vrange = [speed_constraint[attr] for attr in self.var_omega]
        self.var_omega_vlb = np.array([x[0] for x in var_omega_vrange])
        self.var_omega_vub = np.array([x[1] for x in var_omega_vrange])

        self.max_window_size = max_window_size
        self.print_repair_message = print_repair_message

    def _get_dt_seconds(self, t1, t2):
        """Convert time difference to seconds."""
        dt = t2 - t1
        if isinstance(dt, (pd.Timedelta, np.timedelta64)):
            return pd.Timedelta(dt).total_seconds()
        else:
            return float(dt)  # Assume numeric time step

    def _repair_point_forward_backward(self, idx):
        """Repair a point using forward and backward speed constraints."""
        if idx <= 0 or idx >= len(self.dataframe) - 1:
            return

        prev_row = self.dataframe.iloc[idx - 1]
        curr_row = self.dataframe.iloc[idx]
        next_row = self.dataframe.iloc[idx + 1]

        # t_prev = prev_row[self.temporal_attr]
        # t_curr = curr_row[self.temporal_attr]
        # t_next = next_row[self.temporal_attr]

        # dt_fwd = self._get_dt_seconds(t_prev, t_curr)
        # dt_bwd = self._get_dt_seconds(t_curr, t_next)

        dt_fwd = 1
        dt_bwd = 1

        if dt_fwd <= 0 or dt_bwd <= 0:
            return

        # Forward bounds: x_prev + v_lb * dt <= x_curr <= x_prev + v_ub * dt
        lb_fwd = prev_row[self.var_omega].values + dt_fwd * self.var_omega_vlb
        ub_fwd = prev_row[self.var_omega].values + dt_fwd * self.var_omega_vub

        # Backward bounds: x_curr >= x_next - v_ub * dt, x_curr <= x_next - v_lb * dt
        lb_bwd = next_row[self.var_omega].values - dt_bwd * self.var_omega_vub
        ub_bwd = next_row[self.var_omega].values - dt_bwd * self.var_omega_vlb

        # Intersection
        lb = np.maximum(lb_fwd, lb_bwd)
        ub = np.minimum(ub_fwd, ub_bwd)

        # Clip current value to valid range
        curr_vals = curr_row[self.var_omega].values
        repaired = np.clip(curr_vals, lb, ub)

        if np.any(np.abs(repaired - curr_vals) > 1e-6):
            self.dataframe.loc[idx, self.var_omega] = repaired
            if self.print_repair_message:
                print(f"Repaired index {idx}: {curr_vals} -> {repaired}")

    def data_cleaning(self):

        for idx in range(1, len(self.dataframe) - 1):
            self._repair_point_forward_backward(idx)
        
        repaired_data = pd.concat([
            pd.Series(self.timestamps, name=self.temporal_attr),
            self.dataframe.reset_index(drop=True)
        ], axis=1)
        return repaired_data


class Clean4MTSPlus(Clean4MTS):
    """
    Extended cleaner with acceleration constraints.
    """
    def __init__(self, dataframe: pd.DataFrame, speed_constraint: dict, acceleration_constraint: dict,
                 temporal_attr: str, relevant_attrs: list, max_window_size=500, print_repair_message=False):
        super().__init__(dataframe, speed_constraint, temporal_attr, relevant_attrs, max_window_size, print_repair_message)

        # Acceleration constraints
        for attr in acceleration_constraint:
            assert attr in self.var_omega, f"Accel constraint for unknown attr: {attr}"
        var_omega_arange = [acceleration_constraint[attr] for attr in self.var_omega]
        self.var_omega_alb = np.array([x[0] for x in var_omega_arange])
        self.var_omega_aub = np.array([x[1] for x in var_omega_arange])
        assert np.all(self.var_omega_aub > self.var_omega_alb), "Invalid acceleration bounds"

    def _repair_point_with_acceleration(self, idx):
        """Repair a point using speed and acceleration constraints."""
        if idx <= 1 or idx >= len(self.dataframe) - 2:
            return

        df = self.dataframe
        t0 = df.iloc[idx - 2][self.temporal_attr]
        t1 = df.iloc[idx - 1][self.temporal_attr]
        t2 = df.iloc[idx][self.temporal_attr]
        t3 = df.iloc[idx + 1][self.temporal_attr]

        dt01 = self._get_dt_seconds(t0, t1)
        dt12 = self._get_dt_seconds(t1, t2)
        dt23 = self._get_dt_seconds(t2, t3)

        if not all(dt > 0 for dt in [dt01, dt12, dt23]):
            return

        x0 = df.iloc[idx - 2][self.var_omega].values
        x1 = df.iloc[idx - 1][self.var_omega].values
        x2 = df.iloc[idx][self.var_omega].values
        x3 = df.iloc[idx + 1][self.var_omega].values

        # === 1. Speed bounds ===
        lb_fwd = x1 + dt12 * self.var_omega_vlb
        ub_fwd = x1 + dt12 * self.var_omega_vub
        lb_bwd = x3 - dt23 * self.var_omega_vub
        ub_bwd = x3 - dt23 * self.var_omega_vlb
        speed_lb = np.maximum(lb_fwd, lb_bwd)
        speed_ub = np.minimum(ub_fwd, ub_bwd)

        # === 2. Acceleration bounds ===
        # Acceleration: a = (v2 - v1) / dt
        # v1 = (x1 - x0)/dt01, v2 = (x2 - x1)/dt12
        # So: a = ((x2 - x1)/dt12 - (x1 - x0)/dt01) / dt_avg
        # We enforce a in [alb, aub]

        # Approximate time for accel: use average of dt01 and dt12
        dt_acc = (dt01 + dt12) / 2
        if dt_acc <= 0:
            return

        # Max allowed change in velocity
        dv_max = self.var_omega_aub * dt_acc
        dv_min = self.var_omega_alb * dt_acc

        # Predicted velocity from prev segment
        v_prev = (x1 - x0) / dt01
        # Allowed current velocity
        v_curr_min = v_prev + dv_min
        v_curr_max = v_prev + dv_max

        # So x2 must satisfy: x1 + v_curr_min * dt12 <= x2 <= x1 + v_curr_max * dt12
        accel_lb = x1 + v_curr_min * dt12
        accel_ub = x1 + v_curr_max * dt12

        # Also consider backward acceleration (from future)
        v_next = (x3 - x2) / dt23
        v_curr_from_next_min = v_next - self.var_omega_aub * ((dt12 + dt23) / 2)
        v_curr_from_next_max = v_next - self.var_omega_alb * ((dt12 + dt23) / 2)
        # x2 = x1 + v_curr * dt12 => v_curr = (x2 - x1)/dt12
        # So: v_curr >= v_curr_from_next_min => x2 >= x1 + v_curr_from_next_min * dt12
        back_accel_lb = x1 + v_curr_from_next_min * dt12
        back_accel_ub = x1 + v_curr_from_next_max * dt12

        accel_lb = np.maximum(accel_lb, back_accel_lb)
        accel_ub = np.minimum(accel_ub, back_accel_ub)

        # === 3. Combine speed and acceleration bounds ===
        final_lb = np.maximum(speed_lb, accel_lb)
        final_ub = np.minimum(speed_ub, accel_ub)

        # Clip current value
        if np.any(final_lb > final_ub):
            # Conflict, relax to speed only
            final_lb = speed_lb
            final_ub = speed_ub
            if np.any(final_lb > final_ub):
                return  # Unfixable

        repaired = np.clip(x2, final_lb, final_ub)

        if np.any(np.abs(repaired - x2) > 1e-6):
            self.dataframe.loc[idx, self.var_omega] = repaired
            if self.print_repair_message:
                print(f"Repaired index {idx} with accel: {x2} -> {repaired}")

    def data_cleaning(self):

        for idx in range(2, len(self.dataframe) - 2):  # Need context
            self._repair_point_with_acceleration(idx)
        return self.dataframe.copy()


# =================== 示例使用 ===================
if __name__ == '__main__':
    np.random.seed(42)
    n_samples = 100

    # 时间点（数值或时间）
    time_points = np.arange(0, n_samples, 1)  # 或使用 pd.date_range

    # 生成平滑数据
    x1 = np.cumsum(np.random.randn(n_samples) * 0.3)
    x2 = np.sin(np.arange(n_samples) / 5.0) + np.random.randn(n_samples) * 0.05
    y = 0.5 * x1 + 0.3 * x2 + np.random.randn(n_samples) * 0.1

    # 制造一些异常点（突变）
    x1[30] += 5.0
    x2[60] -= 3.0
    y[80] += 4.0

    df = pd.DataFrame({
        'timestamp': time_points,
        'feature1': x1,
        'feature2': x2,
        'target': y
    })

    print("原始数据（前10行）:")
    print(df.head(10))

    # 速度约束（单位/时间步）
    speed_constraints = {
        'feature1': (-1.0, 1.0),
        'feature2': (-0.5, 0.5),
        'target': (-1.0, 1.0)
    }

    # 加速度约束（单位/时间步²）
    acceleration_constraints = {
        'feature1': (-0.5, 0.5),
        'feature2': (-0.2, 0.2),
        'target': (-0.4, 0.4)
    }

    # 创建清洁器（仅速度）
    cleaner = Clean4MTS(
        dataframe=df.copy(),
        speed_constraint=speed_constraints,
        temporal_attr='timestamp',
        relevant_attrs=['feature1', 'feature2', 'target'],
        print_repair_message=False
    )

    cleaned_df_speed = cleaner.data_cleaning()

    # 创建清洁器（速度+加速度）
    cleaner_plus = Clean4MTSPlus(
        dataframe=df.copy(),
        speed_constraint=speed_constraints,
        acceleration_constraint=acceleration_constraints,
        temporal_attr='timestamp',
        relevant_attrs=['feature1', 'feature2', 'target'],
        print_repair_message=False
    )

    cleaned_df_accel = cleaner_plus.data_cleaning()

    print("\n原始数据在 index 30:")
    print(df.iloc[30])
    print("\n仅速度修复后:")
    print(cleaned_df_speed.iloc[30])
    print("\n速度+加速度修复后:")
    print(cleaned_df_accel.iloc[30])