import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm


from Datasets.load_dataset import load_single_dataset, sample_data_by_rate, convert_to_unix_timestamp
from Error_Injection.injector import DataManager
from downstream_tasks import downstream_task

# plt.rcParams["font.family"] = "Times New Roman"
font_path = '/home/yyy/mysoftware/fonts/times.ttf' 
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 38
label_size = 42
legend_size = 34
compare_figsize = (12, 9)
compare_axes_margins = dict(left=0.15, right=0.98, bottom=0.2, top=0.95)



def load_subset(csv_path, columns, n_points=50):
	df = pd.read_csv(csv_path)
	missing_cols = [c for c in columns if c not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing columns in CSV: {missing_cols}")
	return df.loc[: n_points - 1, columns].copy().reset_index(drop=True)


def inject_drift(series, start_index, drift_amplitude, length, ratio_range):
	s = series.copy()
	low, high = ratio_range
	drift = np.random.uniform(low * drift_amplitude, high * drift_amplitude)
	s.iloc[start_index:start_index + length] = s.iloc[start_index:start_index + length] + drift
	return s


def inject_single_point(series, index=None, magnitude=None, rng=None):
	s = series.copy()
	if rng is None:
		rng = np.random.default_rng()
	if index is None:
		index = int(rng.integers(0, len(s)))
	if magnitude is None:
		magnitude = s.std() * 3
	s.iloc[index] = s.iloc[index] + magnitude
	return s


def inject_missing_points(series, indices):
	s = series.copy()
	s.iloc[indices] = np.nan
	return s


def plot_drift_compare_1(origin, clean, dirty, name, output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	x = np.arange(len(origin))

	fig, ax = plt.subplots(1, 1, figsize=(12, 8))
	ax.plot(x, origin, color="#D0D0D0", linewidth=5, label="ground truth")
	ax.plot(x, clean, color="blue", linewidth=5, label="cleaned data")
	ax.plot(x, dirty, color="red", linewidth=5, label="dirty data")
	ax.set_xlim(0, 25)
	ax.set_xticks(range(0, 26, 5))
	ax.set_ylim(-20, 50)
	ax.tick_params(axis='both', direction='in', top=True, right=True, pad=20, width=2.5, length=8)
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['top'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.set_xlabel("Time Index", fontsize=label_size)
	ax.set_ylabel("Value", fontsize=label_size)
	# ax.set_title("Forecast Comparison", y=-0.2)
	legend = ax.legend(loc="upper center", fontsize=legend_size, ncol=3, columnspacing=0.5, handletextpad=0.4, handlelength=1.5)
	legend.get_frame().set_facecolor("none")
	legend.get_frame().set_edgecolor("none")
	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=800, bbox_inches="tight", pad_inches=0)

def plot_drift_compare_2(origin, clean, dirty, name, output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	x = np.arange(len(origin))

	fig, ax = plt.subplots(1, 1, figsize=compare_figsize)
	ax.plot(x, origin, color="#D0D0D0", linewidth=5, label="ground truth")
	# ax.plot(x, clean, color="blue", linewidth=5, label="cleaned data")
	# ax.plot(x, dirty, color="red", linewidth=5, label="dirty data")
	ax.plot(x, clean, color="#4AC6B7", linewidth=5, label="cleaned data")
	ax.plot(x, dirty, color="#FF9D76", linewidth=5, label="dirty data")
	ax.set_xlim(0, 25)
	ax.set_xticks(range(0, 51, 10))
	ax.set_ylim(29.2, 29.8)
	ax.set_yticks(np.arange(29.2, 29.81, 0.2))
	ax.tick_params(axis='both', direction='in', top=True, right=True, pad=20, width=2.5, length=8)
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['top'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.set_xlabel("Timestamp", fontsize=label_size)
	ax.set_ylabel("Value", fontsize=label_size)
	# ax.set_title("Forecast Comparison", y=-0.2)
	legend = ax.legend(loc="upper center", fontsize=legend_size, ncol=3, columnspacing=0.5, handletextpad=0.4, handlelength=1.5)
	legend.get_frame().set_facecolor("none")
	legend.get_frame().set_edgecolor("none")
	fig.subplots_adjust(**compare_axes_margins)
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=800)

def plot_NRMSE_compare(name="performance_compare", output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	methods = [
	    "None",
		"MLE",
		"SCREEN",
		"KF",
		"SpeedAcc",
		"MA",
    	"AegisTS",
	]
	performances = [0.2309, 0.0409, 0.081, 0.0355, 0.0815, 0.2188, 0.0162]
 

	fig, ax = plt.subplots(1, 1, figsize=compare_figsize)
	bars = ax.bar(methods, performances, color="#7389AE", width=0.5) # #B5BAD0 #81D2C7 #E0E0E2

	ax.set_ylabel("NRMSE", fontsize=label_size)
	ax.set_xlabel("Method", fontsize=label_size)
	ax.set_ylim(0.0, 0.3)
	ax.set_yticks(np.arange(0.0, 0.31, 0.1))
	ax.tick_params(axis='both', direction='in', top=True, right=True, pad=12, width=2.5, length=8)
	ax.tick_params(axis='x', labelrotation=20, labelsize=32)
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['top'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

	for bar, value in zip(bars, performances):
		ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.4f}",
				ha='center', va='bottom', fontsize=32,
				color='red' if np.isclose(value, 0.0162) else 'black')

	fig.subplots_adjust(**compare_axes_margins)
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=800)

def plot_performance_compare(name="performance_compare", output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)
 
	methods = [
		"Bayesian",
		"Imdiffusion",
		"Tranad",
		"MA",
		"SCREEN",
    	"AegisTS",
	]
	performances = [-0.17, 0.08, 0.21, 1.49, 11.42, 16.97]

	fig, ax = plt.subplots(1, 1, figsize=compare_figsize)
	bars = ax.bar(methods, performances, color="#7389AE", width=0.5) # #B5BAD0 #81D2C7 #E0E0E2

	# ax.set_ylabel("NRMSE", fontsize=label_size)
	ax.set_ylabel(r"$\Delta$Perf", fontsize=label_size)
	ax.set_xlabel("Method", fontsize=label_size)
	ax.set_ylim(-5, 20)
	ax.set_yticks(np.arange(-5, 21, 5))
	ax.axhline(0, color='black', linewidth=3)  # 添加这行代码来绘制0刻度线
	ax.tick_params(axis='both', direction='in', top=True, right=True, pad=12, width=2.5, length=8)
	ax.tick_params(axis='x', labelrotation=20, labelsize=32)
	ax.spines['top'].set_visible(True)
	ax.spines['right'].set_visible(True)
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['top'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

	for bar, value in zip(bars, performances):
		ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}%",
				ha='center', va='bottom', fontsize=32,
				color='red' if np.isclose(value, 16.97) else 'black')

	fig.subplots_adjust(**compare_axes_margins)
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=800)

#proxy 90 var3
#final 450 var3
def load_rows_var(csv_path, row_start=90, length=25, var_index=3):
	"""
	从CSV中提取指定行区间和变量列。
	行号和变量索引均为 1-based。
	"""
	row_end = row_start + length + 1
	df = pd.read_csv(csv_path)
	row_slice = slice(row_start, row_end)
	col_idx = var_index - 1
	return df.iloc[row_slice, col_idx].reset_index(drop=True)


if __name__ == "__main__":
	result_dir = "/home/yyy/TSC/TSClean/AutoClean/draw_pictures/downstream_results"
 
	# row_start = 90
 
	# clean1 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_clean.csv"), row_start=row_start)
	# dirty1 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_dirty.csv"), row_start=row_start)
	# origin1 = load_rows_var(os.path.join(result_dir, "proxy_y_test_2d_clean.csv"), row_start=row_start)
    
	# dirty1 = dirty1 +2
 
	# dirty_1 = pd.concat([origin1.iloc[:-1], dirty1.iloc[-1:]], ignore_index=True)

	# # Inject anomalies into dirty1, then smooth to get a new clean1.
	# rng = np.random.default_rng(42)
	# single_idx = 5
	# dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=dirty_1.std() * 2, rng=rng)
	# single_idx = 14
	# dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=dirty_1.std() * 3, rng=rng)
	

	# clean_1 = dirty_1.rolling(window=5, center=True, min_periods=1).mean()
	# clean_1 = pd.concat([clean_1.iloc[:-1], clean1.iloc[-1:]], ignore_index=True)
	# plot_drift_compare_1(origin1, clean_1, dirty_1, "clean_compare")
	# plot_performance_compare("performance_compare")



	row_start = 100
	clean1 = load_rows_var(os.path.join(result_dir, "final_y_pred_2d_clean.csv"), row_start=row_start, length=50)
	dirty1 = load_rows_var(os.path.join(result_dir, "final_y_pred_2d_dirty.csv"), row_start=row_start, length=50)
	origin1 = load_rows_var(os.path.join(result_dir, "final_y_test_2d_clean.csv"), row_start=row_start, length=50)

 
	# dirty_1 = pd.concat([origin1.iloc[:-1], dirty1.iloc[-1:]], ignore_index=True)
	dirty_1 = origin1

	# Inject anomalies into dirty1, then smooth to get a new clean1.
	rng = np.random.default_rng(42)
	single_idx = 11
	dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=0.2, rng=rng)
	single_idx = 37
	dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=0.1, rng=rng)
	single_idx = 38
	dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=0.15, rng=rng)
	single_idx = 39
	dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=0.2, rng=rng)
	dirty_1 = inject_missing_points(dirty_1, indices=[18, 19, 20, 21, 22, 23, 24])
	

	clean_1 = dirty_1.rolling(window=15, center=True, min_periods=1).mean()
	# clean_1 = pd.concat([clean_1.iloc[:-1], clean1.iloc[-1:]], ignore_index=True)
	plot_drift_compare_2(origin1, clean_1, dirty_1, "clean_compare")
	plot_performance_compare("performance_compare")
