from Error_Cleaner.tools.constraint_violation_handlers.waste.TwoMILP import TwoMILP
from Error_Cleaner.tools.constraint_violation_handlers.SpeedAcc import SpeedAcc
from Error_Cleaner.tools.constraint_violation_handlers.waste.RCSWS_ import RCSWS
from Error_Cleaner.tools.constraint_violation_handlers.Screen import SCREEN
from Error_Cleaner.tools.constraint_violation_handlers.waste.QCQP_ import QCQP
from Error_Cleaner.tools.constraint_violation_handlers.OneMILP import OneMILP
from Error_Cleaner.tools.constraint_violation_handlers.MTSClean_liner import MTSClean
from Error_Cleaner.tools.constraint_violation_handlers.MTCSC_AS import mtcsc_as_denoise_multispeed
from Error_Cleaner.tools.constraint_violation_handlers.MTCSC_N import MTCSC_N
from Error_Cleaner.tools.constraint_violation_handlers.waste.LsGreedy import Lsgreedy
from Error_Cleaner.tools.constraint_violation_handlers.HTD import HTD
from Error_Cleaner.tools.constraint_violation_handlers.waste.HoloClean import HoloClean
from Error_Cleaner.tools.constraint_violation_handlers.waste.Clean4MTS_ import Clean4MTS
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import speed_plus_acceleration_constraint_clean_local,speed_plus_acceleration_constraint_clean_global,speed_constraint_clean_local,speed_constraint_clean_global,variance_constraint_clean



import pandas as pd
import numpy as np
from Error_Detection.miner import mine_all_constraints,constraint_report


def inject_anomalies(df):
    """
    根据用户指定的 8 个异常位置注入异常。
    返回: new_df (copy), anomaly_log DataFrame
    """
    df2 = df.copy().reset_index(drop=True)

    df2['timestamp'] = pd.to_datetime(df2['timestamp'])

    # Prepare anomaly specifications: (timestamp_str, {col: new_value}, reason_tag)
    anomalies = [
        # 1: S - temperature sudden jump at 2025-10-29 09:00 -> temperature 40.0
        ('2025-10-29 09:00:00', {'temperature': 40.0}, 'S: temp sudden jump'),
        # 2: A - temperature bad second-difference at 2025-10-29 10:00 -> temperature 20.0
        ('2025-10-29 10:00:00', {'temperature': 20.0}, 'A: temp accel anomaly'),
        # 3: R - night temperature sudden rise at 2025-10-29 03:00 -> temperature 25.0
        ('2025-10-29 03:00:00', {'temperature': 25.0}, 'R: night trend violation'),
        # 4: C - flow up but pressure down at 2025-10-29 08:30 -> pressure 95 (flow keep)
        ('2025-10-29 08:30:00', {'pressure': 95.0}, 'C: cross-var (pressure dropped when flow up)'),
        # 5: L - pressure out of logical range at 2025-10-29 14:00 -> pressure 150
        ('2025-10-29 14:00:00', {'pressure': 150.0}, 'L: pressure out-of-range'),
        # 6: L - temperature too low at 2025-10-30 02:00 -> temperature 10.0
        ('2025-10-30 02:00:00', {'temperature': 10.0}, 'L: temp below logical min'),
        # 7: S - flow sudden jump at 2025-10-29 19:30 -> flow 100
        ('2025-10-29 19:30:00', {'flow': 100}, 'S: flow sudden jump'),
        # 8: C - flow up but pressure -5 at 2025-10-29 17:30 -> pressure 95
        ('2025-10-29 17:30:00', {'pressure': 95.0}, 'C: cross-var contradiction')
    ]

    log_rows = []
    for ts_str, changes, reason in anomalies:
        ts = pd.to_datetime(ts_str)
        # find row index where timestamp equals ts
        matches = df2.index[df2['timestamp'] == ts].tolist()
        if not matches:
            # If exact timestamp not found, find nearest timestamp (shouldn't be needed here)
            # but we fallback to nearest by absolute difference
            diffs = (df2['timestamp'] - ts).abs()
            i = int(diffs.idxmin())
        else:
            i = matches[0]
        # Save original values for all affected columns (and full row context)
        orig_row = df2.loc[i].to_dict()
        applied = {}
        for col, new_val in changes.items():
            orig_val = df2.at[i, col]
            df2.at[i, col] = new_val
            applied[col] = {'old': orig_val, 'new': new_val}
        log_rows.append({
            'index': i,
            'timestamp': df2.at[i, 'timestamp'],
            'applied_changes': applied,
            'reason': reason,
            'orig_row_snapshot': orig_row
        })

    log_df = pd.DataFrame(log_rows)
    return df2, log_df



if __name__ == "__main__":

    from Error_Injection.injector import DataManager

    # dm = DataManager(
    #     "idf", "/home/yyy/TSC/TSClean/AutoClean/Datasets/IDF_Power/IDF_Power_Clean.csv"
    # )
    # dm.inject_errors(
    #     0.3,
    #     ["missing", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
    #     covered_attrs=dm.clean_data.columns,
    # )

    # data_abnormal = dm.restored_observed_data
    # label = dm.error_mask.any(axis=1).astype(int)
    # print(label.unique())


    # 1) create clean ts
    df_clean = pd.read_csv("data_test.csv")
    # 2) inject anomalies
    df_dirty, _ = inject_anomalies(df_clean)

    # print("测试数据",df_dirty)

    constraints = mine_all_constraints(df_dirty)
    # print(constraints)

    row_constraints = constraints['row_constraints']
    speed_constraints = constraints['speed_constraints']
    accel_constraints = constraints['accel_constraints']
    variance_constraints = constraints['variance_constraints']
    od_constraints = constraints['od_constraints']
    dd_constraints = constraints['dd_constraints']  
    td_constraints = constraints['td_constraints']

    print(row_constraints)
    print(speed_constraints)
    print(accel_constraints)
    print(variance_constraints)
    print(od_constraints)
    print(dd_constraints)
    print(td_constraints)


    methods = [SCREEN,]


    results = {}
    for func in methods:
        print(f"\n 运行 {func.__name__} ...")
        try:
            data_repaired = func(df_dirty)
            if not isinstance(data_repaired, pd.DataFrame):
                raise ValueError("返回类型错误，必须是 DataFrame")
            # 简单的效果验证：修复点的数值波动应减小
            diff = np.mean(np.abs(data_repaired - df_dirty))
            results[func.__name__] = diff
            print(f"{func.__name__} 完成，平均改变量: {diff:.4f}")
        except Exception as e:
            print(f"{func.__name__} 失败: {e}")
            results[func.__name__] = None

    print("\n=== 测试结果汇总 ===")
    for name, score in results.items():
        print(f"{name:35s} => {'Fail' if score is None else f'{score:.4f}'}")