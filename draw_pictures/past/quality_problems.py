import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm

# plt.rcParams["font.family"] = "Times New Roman"
font_path = '/home/yyy/mysoftware/fonts/times.ttf' 
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 36
label_size = 40
legend_size = 30

def load_subset(start_point, n_points, HULL, MULL):
    
    HULL_ = HULL.loc[start_point: start_point+n_points].copy().reset_index(drop=True)
    MULL_ = MULL.loc[start_point: start_point+n_points].copy().reset_index(drop=True)

    return HULL_, MULL_


def apply_missing(series, missing_indices):
    s = series.copy()
    s.iloc[missing_indices] = np.nan
    return s


def inject_single_point(series, index, magnitude):
    s = series.copy()
    s.iloc[index] = s.iloc[index] + magnitude
    return s


def inject_drift(series, start_index, drift_amplitude, length, ratio_range=(0.3, 0.4)):
    s = series.copy()
    low, high = ratio_range
    drift = np.random.uniform(low * drift_amplitude, high * drift_amplitude)
    s.iloc[start_index:start_index + length] = s.iloc[start_index:start_index + length] - drift
    return s


def inject_speed_violation(series, index_list, max_delta, scale_list):
    s = series.copy()
    for i, index in enumerate(index_list):
        s.iloc[index] = s.iloc[index - 1] + abs(max_delta) * scale_list[i]
    return s


def remove_zero_tick(ax):
    def _formatter(value, _pos):
        if abs(value) < 1e-9:
            return ""
        if abs(value - int(value)) < 1e-9:
            return f"{int(value)}"
        return f"{value:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_formatter))


def _format_axes(ax, n_points):
    ax.set_xlim(0, n_points)
    ax.set_xticks(range(0, n_points + 1, 10))
    ax.set_ylim(0, 10)
    ax.tick_params(axis='both', direction='in', top=True, right=True, pad=20)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    remove_zero_tick(ax)


def plot_quality_missing(x, HULL, MULL, n_points, output_dir):
    missing_idx_HULL = [20, 21, 22, 23, 24, 25, 26]
    missing_idx_MULL = [40, 42, 43, 44, 45, 46, 47]
    HULL_missing = apply_missing(HULL, missing_idx_HULL)
    MULL_missing = apply_missing(MULL, missing_idx_MULL)

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.plot(x, HULL_missing, color="#1f77b4", linewidth=3, label="HULL")
    ax.plot(x, MULL_missing, color="#ff7f0e", linewidth=3, label="MULL")

    def connect_missing_ranges(series_missing, missing_indices, color):
        if not missing_indices:
            return
        groups = []
        start_group = [missing_indices[0]]
        for i in range(1, len(missing_indices)):
            if missing_indices[i] == missing_indices[i - 1] + 1:
                start_group.append(missing_indices[i])
            else:
                groups.append(start_group)
                start_group = [missing_indices[i]]
        groups.append(start_group)

        for group in groups:
            before_idx = group[0] - 1
            after_idx = group[-1] + 1
            if 0 <= before_idx < len(x) and 0 < after_idx < len(x):
                y_before = series_missing.iloc[before_idx]
                y_after = series_missing.iloc[after_idx]
                ax.plot([before_idx, after_idx], [y_before, y_after], color=color, linestyle="--", linewidth=3, label="" if group == groups[0] else "missing data")

    connect_missing_ranges(HULL_missing, missing_idx_HULL, "#B0B0B0")
    connect_missing_ranges(MULL_missing, missing_idx_MULL, "#B0B0B0")

    _format_axes(ax, n_points)
    ax.set_xlabel("Time Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "quality_missing.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_quality_outliers(x, HULL, MULL, n_points, output_dir):
    single_index = 23
    single_magnitude = 5.0 * float(HULL.std())
    drift_start = 19
    drift_amplitude = 4.0 * float(MULL.std())
    drift_length = 5
    HULL_anomaly = inject_single_point(HULL, single_index, single_magnitude)
    MULL_anomaly = inject_drift(MULL, drift_start, drift_amplitude, drift_length)

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.plot(x, HULL_anomaly, color="#1f77b4", linewidth=3, label="dirty HULL", zorder=3)
    ax.plot(x, HULL, color="#B0B0B0", linewidth=3)
    ax.scatter([single_index], [HULL_anomaly.iloc[single_index]], color="red", marker="o", s=60, zorder=5)
    ax.plot(x, MULL_anomaly, color="#ff7f0e", linewidth=3, label="dirty MULL", zorder=3)
    ax.plot(x, MULL, color="#D0D0D0", linewidth=3, label="clean data")
    ax.scatter(x[drift_start:drift_start + drift_length], [MULL_anomaly.iloc[i] for i in range(drift_start, drift_start + drift_length)], color="red", marker="o", s=60, zorder=5, label="outliers")

    _format_axes(ax, n_points)
    ax.set_xlabel("Time Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "quality_outliers.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_quality_constraint(x, HULL, MULL, n_points, output_dir):
    HULL_diff = np.abs(np.diff(HULL.to_numpy()))
    MULL_diff = np.abs(np.diff(MULL.to_numpy()))
    HULL_max_delta = np.max(HULL_diff)
    MULL_max_delta = np.max(MULL_diff)

    HULL_violate = inject_speed_violation(HULL, [35, 36], HULL_max_delta, [1.3, 0.3])
    MULL_violate = inject_speed_violation(MULL, [44, 45], MULL_max_delta, [-1.2, 0.5])

    HULL_viol_idx = np.where(np.abs(np.diff(HULL_violate.to_numpy())) > HULL_max_delta)[0] + 1
    MULL_viol_idx = np.where(np.abs(np.diff(MULL_violate.to_numpy())) > MULL_max_delta)[0] + 1

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.plot(x, HULL_violate, color="#1f77b4", linewidth=3, label="dirty HULL", zorder=3)
    ax.plot(x, HULL, color="#B0B0B0", linewidth=3)
    ax.scatter(HULL_viol_idx, HULL_violate.iloc[HULL_viol_idx], color="#d62728", marker="o", s=60, zorder=5)
    ax.plot(x, MULL_violate, color="#ff7f0e", linewidth=3, label="dirty MULL", zorder=3)
    ax.plot(x, MULL, color="#D0D0D0", linewidth=3, label="clean data")
    ax.scatter(MULL_viol_idx, MULL_violate.iloc[MULL_viol_idx], color="#d62728", marker="o", s=60, zorder=5, label="Violation")

    _format_axes(ax, n_points)
    ax.set_xlabel("Time Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)
    ax.legend(loc="upper left", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "quality_constraint.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_quality_problems(csv_path, output_dir=None, n_points=50):
    df = pd.read_csv(csv_path)
    HULL = df["HULL"]
    MULL = df["MULL"]
    output_dir = output_dir or os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(n_points + 1)
    # data = df.loc[start_point: start_point+n_points - 1, columns].copy().reset_index(drop=True)
    HULL1, MULL1 = load_subset(50, n_points, HULL, MULL)
    HULL2, MULL2 = load_subset(0, n_points, HULL, MULL)
    HULL3, MULL3 = load_subset(230, n_points, HULL, MULL)

    plot_quality_missing(x, HULL1, MULL1, n_points, output_dir)
    plot_quality_outliers(x, HULL2, MULL2, n_points, output_dir)
    plot_quality_constraint(x, HULL3, MULL3, n_points, output_dir)


if __name__ == "__main__":
    csv_path = "/home/yyy/TSC/TSClean/AutoClean/draw_pictures/ETTh1.csv"
    plot_quality_problems(csv_path)