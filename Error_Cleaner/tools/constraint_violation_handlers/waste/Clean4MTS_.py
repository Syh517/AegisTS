import numpy as np
import pandas as pd
from collections import deque
from scipy.optimize import minimize


class OptimizeProblemPlus:
    """
    Optimization problem for data repair.
    Minimizes a cost function that balances:
    - Deviation from regression rule
    - Movement cost (weighted by attribute scale)
    - Number of changed attributes
    """
    def __init__(self, data_instance: np.ndarray, lb: np.ndarray, ub: np.ndarray, rr, beta=1, w1=None, w2=None,
                 repair_attr=None):
        self.datainstance = np.array(data_instance).flatten()
        self.Dim = len(self.datainstance)
        self.rr = rr  # [model_info, tolerance, bias_shift]
        self.beta = beta
        self.w1 = w1 if w1 is not None else self.Dim * 3
        self.w2 = w2 if w2 is not None else self.Dim

        # Determine which attributes are allowed to be repaired
        if repair_attr is None:
            self.repair_attr = np.arange(self.Dim)
            self.fixed_attr = np.array([], dtype=int)
        else:
            self.repair_attr = np.array(repair_attr, dtype=int)
            self.fixed_attr = np.setdiff1d(np.arange(self.Dim), self.repair_attr)

        # Bounds are relative to datainstance
        self.lb = lb - self.datainstance
        self.ub = ub - self.datainstance

        # Use only repairable bounds
        self.lb_repair = self.lb[self.repair_attr]
        self.ub_repair = self.ub[self.repair_attr]

        # Compute weights for movement penalty
        w_vals = []
        for i in self.repair_attr:
            low = min(self.lb[i], 0)
            high = max(self.ub[i], 0)
            if high > low:
                w_vals.append(1.0 / (high - low))
            else:
                w_vals.append(0.0)
        self.w = np.array(w_vals).reshape(-1, 1)

        # Prepare scipy bounds
        self.scipy_bounds = [(float(l), float(u)) for l, u in zip(self.lb_repair, self.ub_repair)]

    def objective(self, delta):
        """
        Objective function for scipy.optimize.
        delta: 1D array of changes to repairable attributes.
        """
        delta = np.array(delta).flatten()
        if delta.size != len(self.repair_attr):
            raise ValueError("delta size mismatch")

        # Reconstruct full delta vector
        full_delta = np.zeros(self.Dim)
        full_delta[self.repair_attr] = delta

        # Apply delta to data
        x_repaired = self.datainstance + full_delta
        src = x_repaired[:-1].reshape(1, -1)
        target = x_repaired[-1]

        # Predict using regression rule
        try:
            if self.rr[0][0] == 'bayesian':
                pred = self.rr[0][1].predict(np.vander(src.T[0], N=4))
            else:
                pred = self.rr[0][1].predict(src)
            bias = (target - pred.flatten())[0] + self.rr[2]
            deg = abs(bias) - self.rr[1]
        except Exception:
            deg = np.inf

        # Degree penalty
        if deg <= 0:
            deg_penalty = deg / self.rr[1]
        else:
            deg_penalty = (1 - np.exp(-deg / self.rr[1] * self.beta)) + self.w1 + self.w2

        # Movement penalty
        movement_penalty = np.dot(np.abs(delta), self.w.flatten()) * self.w1 / len(delta)

        # Count penalty: how many attributes changed significantly?
        small_bound = (self.ub_repair - self.lb_repair) / 1000
        count = np.sum(np.abs(delta) > small_bound) / len(delta)
        count_penalty = self.w2 * count

        return movement_penalty + count_penalty + deg_penalty


def solve_optimization(problem: OptimizeProblemPlus, prophet, method=None, NIND=50, MAXGEN=500, trappedValue=1e-6):
    """
    Solve optimization using scipy.optimize.minimize (SLSQP).
    Uses prophet points to initialize.
    """
    # Convert prophet to delta space
    if prophet is None or len(prophet) == 0:
        init_delta = np.zeros(len(problem.repair_attr))
    else:
        try:
            deltas = []
            for p in prophet:
                p = np.array(p).flatten()
                if p.shape[0] == problem.Dim:
                    d = p[problem.repair_attr] - problem.datainstance[problem.repair_attr]
                else:
                    d = p - problem.datainstance[problem.repair_attr]
                deltas.append(d)
            scores = [problem.objective(d) for d in deltas]
            best_idx = np.argmin(scores)
            init_delta = deltas[best_idx]
        except Exception:
            init_delta = np.zeros(len(problem.repair_attr))

    # Run optimization
    try:
        res = minimize(
            fun=problem.objective,
            x0=init_delta,
            bounds=problem.scipy_bounds,
            method='SLSQP',
            options={'maxiter': MAXGEN, 'ftol': trappedValue, 'disp': False}
        )
        final_delta = res.x if res.success else init_delta
    except Exception:
        final_delta = init_delta

    # Reconstruct result
    result = problem.datainstance.copy()
    result[problem.repair_attr] += final_delta
    return result


class Clean4MTS:
    def __init__(self, dataframe: pd.DataFrame, regression_rule: list, speed_constraint: dict, temporal_attr: str,
                 src: list, target: list, delta_w=12, beta=0.5, max_window_size=500, print_repair_message=False,
                 method='studGA', NIND=50, MAXGEN=500, trappedValue=1e-6, w1=None, w2=None):
        # Validate inputs
        assert len(src) > 0, "src cannot be empty"
        assert len(target) == 1, "target must contain exactly one column"
        assert temporal_attr in dataframe.columns, f"{temporal_attr} not in dataframe"
        assert all(col in dataframe.columns for col in src + target), "Some columns not in dataframe"

        # Select relevant columns
        attrs = list(set(src + target + [temporal_attr]))
        self.dataframe = dataframe[attrs].reset_index(drop=True)
        self.var_omega = src + target
        assert temporal_attr not in self.var_omega, "temporal_attr should not be in src or target"

        # Validate speed constraints
        for attr in speed_constraint:
            assert attr in self.var_omega, f"Speed constraint for unknown attr: {attr}"
        self.sc = speed_constraint
        self.rr = regression_rule  # [model, tolerance, bias_shift]
        self.src = src
        self.target = target
        self.temporal_attr = temporal_attr
        self.delta_w = delta_w
        self.beta = beta
        self.vtw_index = []  # Violation time windows
        self.max_window_size = max_window_size
        self.method = method
        self.w1 = w1
        self.w2 = w2
        self.NIND = NIND
        self.MAXGEN = MAXGEN
        self.trappedValue = trappedValue
        self.print_repair_message = print_repair_message

        # Precompute speed bounds
        var_omega_vrange = [speed_constraint[attr] for attr in self.var_omega]
        self.var_omega_vlb = np.array([x[0] for x in var_omega_vrange])
        self.var_omega_vub = np.array([x[1] for x in var_omega_vrange])

    def quantify_deviation_through_sliding_window(self, start_idx, window_size):
        """Compute deviation from regression rule in a window."""
        end_idx = min(start_idx + window_size, len(self.dataframe))
        if end_idx <= start_idx:
            return np.array([])

        xtest = self.dataframe.iloc[start_idx:end_idx][self.src].values
        ytest = self.dataframe.iloc[start_idx:end_idx][self.target].values.flatten()

        try:
            if self.rr[0][0] == 'bayesian':
                pred = self.rr[0][1].predict(np.vander(xtest[:, 0], N=4))
            else:
                pred = self.rr[0][1].predict(xtest)
            bias = ytest - pred + self.rr[2]
            return np.abs(bias) - self.rr[1]
        except Exception as e:
            print(f"Prediction error in window {start_idx}-{end_idx}: {e}")
            return np.zeros(end_idx - start_idx)

    def violation_localization(self, profile):
        """
        Find maximal violation intervals from profile.
        Profile: list of (start, end, score) triples.
        """
        if not profile:
            return

        # Sort by start time
        profile.sort(key=lambda x: x[0])
        stack = []

        for tri in profile:
            s, e, score = tri
            if not stack:
                stack.append(tri)
            else:
                last_s, last_e, last_score = stack[-1]
                if s <= last_e + 1:  # Overlapping or adjacent
                    merged_e = max(last_e, e)
                    merged_score = last_score + score
                    stack[-1] = (last_s, merged_e, merged_score)
                else:
                    stack.append(tri)

        # Add merged intervals
        for s, e, _ in stack:
            self.vtw_index.append((s, e))

    def violation_profiling(self, max_window_size):
        """Identify violation time windows using sliding window."""
        self.vtw_index = []
        index = 0
        profile = []

        while index < len(self.dataframe):
            dev = self.quantify_deviation_through_sliding_window(index, max_window_size)
            if len(dev) == 0:
                index += max_window_size
                continue

            vtis = np.where(dev > 0)[0]  # Indices where deviation > 0
            if len(vtis) == 0:
                index += max_window_size
                continue

            # Process violation segments
            seg_start = vtis[0]
            seg_end = vtis[0]
            segments = []

            for i in range(1, len(vtis)):
                if vtis[i] == vtis[i-1] + 1:
                    seg_end = vtis[i]
                else:
                    segments.append((seg_start, seg_end))
                    seg_start = vtis[i]
                    seg_end = vtis[i]
            segments.append((seg_start, seg_end))

            # Add segments to profile
            for s_idx, e_idx in segments:
                abs_s = index + s_idx
                abs_e = index + e_idx
                score = np.sum(1 - np.exp(-dev[s_idx:e_idx+1] / self.rr[1] * self.beta))
                profile.append((abs_s, abs_e, score))

            index += max_window_size

        # Merge overlapping intervals
        self.violation_localization(profile)

    def prophet_generator(self, data_instance, lb, ub, weight, n=None):
        """Generate candidate repair points around data_instance."""
        dim = len(self.var_omega)
        if n is None:
            n = 3 * dim
        if weight is None:
            weight = np.ones(dim)
        weight = np.array(weight) / (np.sum(weight) + 1e-8)

        prophets = []
        for _ in range(n):
            p = data_instance.copy()
            j = np.random.choice(dim, p=weight)
            # Perturb within bounds
            mid = (lb[j] + ub[j]) / 2
            rng = (ub[j] - lb[j]) / 2
            noise = np.random.normal(0, rng / 3)
            p[j] = mid + noise
            p[j] = np.clip(p[j], lb[j], ub[j])
            prophets.append(p)
        return np.array(prophets)

    def repair_violation_single(self, s, e):
        """Repair a single-point violation (s to e, where e = s+1)."""
        if s <= 0 or e >= len(self.dataframe) - 1:
            if self.print_repair_message:
                print(f"Skip repair at {s} due to boundary")
            return

        p = self.dataframe.iloc[s-1:e+1][self.var_omega].values
        t = self.dataframe.iloc[s-1:e+1][self.temporal_attr].diff().values[1:]

        if len(t) < 2:
            return

        dt_prev, dt_next = t[0], t[1]
        if isinstance(dt_prev, np.timedelta64):
            dt_prev = dt_prev / np.timedelta64(1, 's')
        if isinstance(dt_next, np.timedelta64):
            dt_next = dt_next / np.timedelta64(1, 's')

        # Forward and backward bounds
        lb_fwd = p[0] + dt_prev * self.var_omega_vlb
        ub_fwd = p[0] + dt_prev * self.var_omega_vub
        lb_bwd = p[2] - dt_next * self.var_omega_vub
        ub_bwd = p[2] - dt_next * self.var_omega_vlb

        # Intersection
        lb = np.maximum(lb_fwd, lb_bwd)
        ub = np.minimum(ub_fwd, ub_bwd)
        lb = np.minimum(lb, p[1])  # Ensure valid
        ub = np.maximum(ub, p[1])

        # Check if repair needed
        center = (lb + ub) / 2
        diff = np.abs(p[1] - center) / (ub - lb + 1e-8)
        if np.all(diff <= 0.5):
            return

        # Generate candidates
        prophet = self.prophet_generator(p[1], lb, ub, weight=diff, n=self.NIND - 1)
        prophet = np.vstack([prophet, p[0], p[2]])  # Include neighbors

        op = OptimizeProblemPlus(p[1], lb, ub, self.rr, self.beta, self.w1, self.w2)
        repaired = solve_optimization(op, prophet, MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)

        if self.print_repair_message:
            print(f"Repair index {s}: {p[1]} -> {repaired}")

        self.dataframe.loc[s, self.var_omega] = repaired

    def repair_violation_continuous(self, s, e):
        """Repair a continuous violation window."""
        width = e - s
        if width == 1:
            self.repair_violation_single(s, e)
            return

        if s <= 0 or e >= len(self.dataframe) - 1:
            return

        mid = (s + e) // 2
        p_left = self.dataframe.iloc[s-1:mid+1][self.var_omega].values
        t_left = self.dataframe.iloc[s-1:mid+1][self.temporal_attr].values
        p_right = self.dataframe.iloc[mid:e+1][self.var_omega].values
        t_right = self.dataframe.iloc[mid:e+1][self.temporal_attr].values

        # Repair left half
        for i in range(1, len(p_left)):
            dt = t_left[i] - t_left[i-1]
            lb = p_left[i-1] + dt * self.var_omega_vlb
            ub = p_left[i-1] + dt * self.var_omega_vub
            if i < len(t_left) - 1:
                dt_next = t_left[i+1] - t_left[i]
                lb_bwd = p_left[i+1] - dt_next * self.var_omega_vub
                ub_bwd = p_left[i+1] - dt_next * self.var_omega_vlb
                lb = np.maximum(lb, lb_bwd)
                ub = np.minimum(ub, ub_bwd)

            center = (lb + ub) / 2
            diff = np.abs(p_left[i] - center) / (ub - lb + 1e-8)
            if np.all(diff <= 0.5):
                continue

            prophet = self.prophet_generator(p_left[i], lb, ub, diff, n=self.NIND - 2)
            prophet = np.vstack([prophet, p_left[i-1], p_left[min(i+1, len(p_left)-1)]])
            op = OptimizeProblemPlus(p_left[i], lb, ub, self.rr, self.beta, self.w1, self.w2)
            repaired = solve_optimization(op, prophet, MAXGEN=self.MAXGEN)
            p_left[i] = repaired

        # Repair right half
        for i in range(len(p_right) - 2, -1, -1):
            dt = t_right[i+1] - t_right[i]
            lb = p_right[i] + dt * self.var_omega_vlb
            ub = p_right[i] + dt * self.var_omega_vub
            if i > 0:
                dt_prev = t_right[i] - t_right[i-1]
                lb_fwd = p_right[i-1] + dt_prev * self.var_omega_vlb
                ub_fwd = p_right[i-1] + dt_prev * self.var_omega_vub
                lb = np.maximum(lb, lb_fwd)
                ub = np.minimum(ub, ub_fwd)

            center = (lb + ub) / 2
            diff = np.abs(p_right[i] - center) / (ub - lb + 1e-8)
            if np.all(diff <= 0.5):
                continue

            prophet = self.prophet_generator(p_right[i], lb, ub, diff, n=self.NIND - 2)
            prophet = np.vstack([prophet, p_right[max(i-1, 0)], p_right[i+1]])
            op = OptimizeProblemPlus(p_right[i], lb, ub, self.rr, self.beta, self.w1, self.w2)
            repaired = solve_optimization(op, prophet, MAXGEN=self.MAXGEN)
            p_right[i] = repaired

        # Update dataframe
        self.dataframe.loc[s:mid, self.var_omega] = p_left[1:]
        self.dataframe.loc[mid:e, self.var_omega] = p_right[:-1]

    def data_repair(self):
        """Repair all violation windows."""
        for (s, e) in self.vtw_index:
            if e < s:
                continue
            if e == s:
                self.repair_violation_single(s, s+1)
            else:
                self.repair_violation_continuous(s, e)

    def data_cleaning(self):
        """Main cleaning pipeline."""
        self.violation_profiling(max_window_size=self.max_window_size)
        self.data_repair()
        return self.dataframe.copy()


# Extended class with acceleration constraints
class Clean4MTSPlus(Clean4MTS):
    def __init__(self, dataframe: pd.DataFrame, regression_rule: list, speed_constraint: dict, acceleration_constraint: dict,
                 temporal_attr: str, src: list, target: list, delta_w=6, beta=1, max_window_size=1000,
                 method='studGA', NIND=50, MAXGEN=500, trappedValue=1e-6, w1=None, w2=None, print_repair_message=False):
        super().__init__(dataframe, regression_rule, speed_constraint, temporal_attr, src, target,
                         delta_w, beta, max_window_size, print_repair_message, method, NIND, MAXGEN, trappedValue, w1, w2)

        for attr in acceleration_constraint:
            assert attr in self.var_omega, f"Accel constraint for unknown attr: {attr}"
        var_omega_arange = [acceleration_constraint[attr] for attr in self.var_omega]
        self.var_omega_alb = np.array([x[0] for x in var_omega_arange])
        self.var_omega_aub = np.array([x[1] for x in var_omega_arange])
        assert np.all(self.var_omega_aub > self.var_omega_alb), "Invalid acceleration bounds"

    def repair_violation_single(self, s, e):
        if s <= 1 or e >= len(self.dataframe) - 2:
            return
        # TODO: Implement acceleration-aware repair
        super().repair_violation_single(s, e)

    def repair_violation_continuous(self, s, e):
        # TODO: Extend with acceleration bounds
        super().repair_violation_continuous(s, e)


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression    # 创建示例数据
    np.random.seed(42)
    n_samples = 200

    # 生成时间序列数据
    # time_points = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    time_points = np.arange(0, n_samples, 1)
    # 生成一些特征变量
    x1 = np.cumsum(np.random.randn(n_samples) * 0.5)  # 随机游走
    x2 = np.sin(np.arange(n_samples) / 10) + np.random.randn(n_samples) * 0.1  # 带噪声的正弦波
    # 生成目标变量（与特征有线性关系，加上噪声）
    y = 2 * x1 + 1.5 * x2 + np.random.randn(n_samples) * 0.3

    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': time_points,
        'feature1': x1,
        'feature2': x2,
        'target': y
    })

    print("示例数据:")
    print(df.head(10))

    # 创建回归模型（用于回归规则）
    X_train = df[['feature1', 'feature2']].values
    y_train = df['target'].values

    # 训练一个简单的线性回归模型
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # 定义回归规则 [模型信息, 容忍度, 偏移量]
    regression_rule = [
        ['linear', reg_model],  # 模型类型和模型对象
        0.5,                    # 容忍度
        0.0                     # 偏移量
    ]

    # 定义速度约束 {变量名: (下界, 上界)}
    speed_constraints = {
        'feature1': (-2.0, 2.0),
        'feature2': (-1.0, 1.0),
        'target': (-3.0, 3.0)
    }

    # 创建Clean4MTS实例
    cleaner = Clean4MTS(
        dataframe=df,
        regression_rule=regression_rule,
        speed_constraint=speed_constraints,
        temporal_attr='timestamp',
        src=['feature1', 'feature2'],
        target=['target'],
        delta_w=12,
        beta=0.5,
        max_window_size=100,
        print_repair_message=True,
        NIND=30,
        MAXGEN=100
    )


    print(f"数据形状: {cleaner.dataframe.shape}")
    print(f"变量列表: {cleaner.var_omega}")
    print(f"源变量: {cleaner.src}")
    print(f"目标变量: {cleaner.target}")
    print(f"时间属性: {cleaner.temporal_attr}")

    # 查看速度约束
    print(f"速度约束下界: {cleaner.var_omega_vlb}")
    print(f"速度约束上界: {cleaner.var_omega_vub}")

    # 执行数据清洗过程
    print("\n开始数据清洗...")

    # 显示如何调用清洗方法（注释掉以避免在示例中实际执行）
    cleaned_data =  cleaner.data_cleaning()
    print(df.head(10))