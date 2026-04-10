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
    ax.tick_params(axis='both', direction='in', top=True, right=True, pad=20, width=2.5, length=8)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    remove_zero_tick(ax)


def plot_all_quality_problems(x, HULL, MULL, n_points, output_dir, fig, ax):
    """
    在一张图上绘制缺失值、异常值和约束违规三种质量问题
    """
    # 1. 添加缺失值
    missing_idx_HULL = [2,3,4,5,6]
    missing_idx_MULL = []
    HULL_dirty = apply_missing(HULL, missing_idx_HULL)
    MULL_dirty = apply_missing(MULL, missing_idx_MULL)

    # 2. 添加异常值
    single_index = 20
    single_magnitude = 4.0 * float(HULL.std())
    drift_start = 12
    drift_amplitude = 5.0 * float(MULL.std())
    drift_length = 3
    HULL_dirty = inject_single_point(HULL_dirty, single_index, single_magnitude)
    MULL_dirty = inject_drift(MULL_dirty, drift_start, drift_amplitude, drift_length)

    # 3. 添加约束违规
    HULL_diff = np.abs(np.diff(HULL.to_numpy()))
    MULL_diff = np.abs(np.diff(MULL.to_numpy()))
    HULL_max_delta = np.max(HULL_diff)
    MULL_max_delta = np.max(MULL_diff)

    HULL_dirty = inject_speed_violation(HULL_dirty, [40, 41], HULL_max_delta, [1.1, -0.3])
    MULL_dirty = inject_speed_violation(MULL_dirty, [34, 35], MULL_max_delta, [-1.1, 0.5])

    # 获取约束违规点的位置
    HULL_viol_idx = np.where(np.abs(np.diff(HULL_dirty.to_numpy(), prepend=np.nan)) > HULL_max_delta)[0]
    MULL_viol_idx = np.where(np.abs(np.diff(MULL_dirty.to_numpy(), prepend=np.nan)) > MULL_max_delta)[0]

    # 绘制原始干净数据
    ax.plot(x, HULL, color="#D0D0D0", linewidth=3, label="Clean HULL", linestyle="--", zorder=1)
    ax.plot(x, MULL, color="#D0D0D0", linewidth=3, label="Clean MULL", linestyle="--", zorder=1)

    # 绘制带有所有问题的数据
    ax.plot(x, HULL_dirty, color="#1f77b4", linewidth=3, label="Dirty HULL", zorder=2)
    ax.plot(x, MULL_dirty, color="#0C7779", linewidth=3, label="Dirty MULL", zorder=2)


    # 标记异常值点
    ax.scatter([single_index], [HULL_dirty.iloc[single_index]], color="red", marker="o", s=100, zorder=6, label="Outliers")
    ax.scatter(x[drift_start:drift_start + drift_length], [MULL_dirty.iloc[i] for i in range(drift_start, drift_start + drift_length)], color="red", marker="o", s=100, zorder=6)

    # 标记约束违规点
    if len(HULL_viol_idx) > 0:
        ax.scatter(HULL_viol_idx, HULL_dirty.iloc[HULL_viol_idx], color="#F25912", marker="*", s=200, zorder=5, label="Constraints Violations")
    if len(MULL_viol_idx) > 0:
        ax.scatter(MULL_viol_idx, MULL_dirty.iloc[MULL_viol_idx], color="#F25912", marker="*", s=200, zorder=5)

    _format_axes(ax, n_points)
    ax.set_xlabel("Time Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {}
    for handle, label in zip(handles, labels):
        if label not in label_to_handle:
            label_to_handle[label] = handle

    legend_order = [
        "Clean HULL",
        "Clean MULL",
        "Dirty HULL",
        "Dirty MULL",
        "Outliers",
        "Constraints Violations",
    ]
    ordered_labels = [label for label in legend_order if label in label_to_handle]
    ordered_handles = [label_to_handle[label] for label in ordered_labels]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=legend_size,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(
        os.path.join(output_dir, "all_quality_problems.png"),
        dpi=600,
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
    HULL, MULL = load_subset(230, n_points, HULL, MULL)

    # # 绘制单独的图
    # fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    # plot_quality_missing(x, HULL, MULL, n_points, output_dir, fig, ax)
    
    # fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    # plot_quality_outliers(x, HULL, MULL, n_points, output_dir, fig, ax)
    
    # fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    # plot_quality_constraint(x, HULL, MULL, n_points, output_dir, fig, ax)
    
    # 绘制合并的图
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    plot_all_quality_problems(x, HULL, MULL, n_points, output_dir, fig, ax)


if __name__ == "__main__":
    csv_path = "/home/yyy/TSC/TSClean/AutoClean/draw_pictures/ETTh1.csv"
    plot_quality_problems(csv_path)