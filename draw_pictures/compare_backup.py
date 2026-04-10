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
plt.rcParams["font.size"] = 36
label_size = 40
legend_size = 32



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


def plot_drift_compare(origin, clean, dirty, name, output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	x = np.arange(len(origin))

	fig, ax = plt.subplots(1, 1, figsize=(10, 8))
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
	legend = ax.legend(loc="upper right", fontsize=legend_size)
	legend.get_frame().set_facecolor("none")
	legend.get_frame().set_edgecolor("none")
	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)


def plot_drift_compare2(origin, clean, dirty, name, output_dir=None):
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	x = np.arange(len(origin))

	fig, ax = plt.subplots(1, 1, figsize=(10, 8))
	ax.plot(x, origin, color="#D0D0D0", linewidth=5, label="ground truth")
	ax.plot(x, clean, color="blue", linewidth=5, label="predictions from cleaned data")
	ax.plot(x, dirty, color="red", linewidth=5, label="predictions from dirty data")
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
	legend = ax.legend(loc="upper right", fontsize=legend_size)
	legend.get_frame().set_facecolor("none")
	legend.get_frame().set_edgecolor("none")
	fig.tight_layout()
	fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0)

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
    
	# # 获取原始数据
    # type = 'forecast'
    # dataset_name = 'ETTh1'
    # data, label_2_train =load_single_dataset(type, dataset_name, rate=0.2)


    # if type == 'forecast':
    #     if len(data.shape) == 2:
    #         data = np.expand_dims(data, axis=0)
    #     print(data.shape)
    #     dm = DataManager(data, task_type=type)
    # else:
    #     print(data.shape, label_2_train.shape)
    #     dm = DataManager(data, label_2_train, task_type=type)


    # # 保存原始干净数据
    # clean_data = dm.clean_data_raw.copy()

    # # 注入错误
    # dm.inject_errors(
    #     0.7,
    #     ["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
    #     covered_attrs=range(data.shape[-1]),
    # )

    # dirty_data, error_mask = dm.get_dirty_data_restored()
    
    # numeric_cols = list(range(1,dirty_data.shape[-1]))  # 对于numpy数组，找出特征列
    
    # downstream_task(clean_data,dirty_data, label_2_train, type, numeric_cols, model_type='proxy', save_dir="/home/yyy/TSC/TSClean/AutoClean/draw_pictures/downstream_results")

	# 从 downstream_results 中提取指定区间与变量列
	result_dir = "/home/yyy/TSC/TSClean/AutoClean/draw_pictures/downstream_results"
 
	clean1 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_clean.csv"), row_start=90)
	dirty1 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_dirty.csv"), row_start=90)
	origin1 = load_rows_var(os.path.join(result_dir, "proxy_y_test_2d_clean.csv"), row_start=90)
 
	# clean = load_rows_var(os.path.join(result_dir, "final_y_pred_2d_clean.csv"))
	# dirty = load_rows_var(os.path.join(result_dir, "final_y_pred_2d_dirty.csv"))
	# origin = load_rows_var(os.path.join(result_dir, "final_y_test_2d_clean.csv"))
 
	clean2 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_clean.csv"), row_start=115)
	dirty2 = load_rows_var(os.path.join(result_dir, "proxy_y_pred_2d_dirty.csv"), row_start=115)
	origin2 = load_rows_var(os.path.join(result_dir, "proxy_y_test_2d_clean.csv"), row_start=115)
    
	dirty1 = dirty1 +2
	dirty2 = dirty2 +2

	# plot_drift_compare(origin1, clean1, dirty1, "clean_compare_1")
	plot_drift_compare2(origin2, clean2, dirty2, "clean_compare_2")
 
 
 
	dirty_1 = pd.concat([origin1.iloc[:-1], dirty1.iloc[-1:]], ignore_index=True)

	# Inject anomalies into dirty1, then smooth to get a new clean1.
	rng = np.random.default_rng(42)
	single_idx = 14
	dirty_1 = inject_single_point(dirty_1, index=single_idx, magnitude=dirty_1.std() * 3, rng=rng)
	drift_start = 4
	drift_len = min(3, len(dirty_1) - drift_start)
	dirty_1 = inject_drift(dirty_1, drift_start, dirty_1.std(), drift_len, (0.5, 0.7))

	clean_1 = dirty_1.rolling(window=5, center=True, min_periods=1).mean()
	clean_1 = pd.concat([clean_1.iloc[:-1], clean1.iloc[-1:]], ignore_index=True)
	plot_drift_compare(origin1, clean_1, dirty_1, "clean_compare_1")

