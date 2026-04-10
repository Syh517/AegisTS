import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Datasets.load_dataset import load_single_dataset
from Error_Injection.injector import DataManager
from Error_Detection.Detector import Detector
from Error_Cleaner.parameters import (
    load_original_data,
    main as hrl_main,
    set_lambda_weights,
)
from Error_Cleaner.EvaluationMetrics import evaluate_cleaning_effectiveness


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError(f"Lambda weights must be non-negative, got {weights.tolist()}")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"Sum of lambda weights must be > 0, got {total}")
    return weights / total


def generate_grid_candidates(
    step: float = 0.1,
    lambda_min: float = 0.1,
    lambda_max: float = 0.8,
) -> List[Tuple[float, float, float]]:
    if step <= 0:
        raise ValueError(f"grid_step must be > 0, got {step}")

    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError(f"grid_step={step} 无法整除 1，请使用 0.5/0.25/0.2/0.1 等")

    min_units = int(round(lambda_min / step))
    max_units = int(round(lambda_max / step))
    if not np.isclose(min_units * step, lambda_min):
        raise ValueError(f"lambda_min={lambda_min} 与 step={step} 不兼容")
    if not np.isclose(max_units * step, lambda_max):
        raise ValueError(f"lambda_max={lambda_max} 与 step={step} 不兼容")
    if min_units > max_units:
        raise ValueError(f"lambda_min={lambda_min} 不能大于 lambda_max={lambda_max}")

    candidates = []
    for a in range(min_units, max_units + 1):
        for b in range(min_units, max_units + 1):
            c = units - a - b
            if c < min_units or c > max_units:
                continue
            l1 = round(a / units, 10)
            l2 = round(b / units, 10)
            l3 = round(c / units, 10)
            candidates.append((float(l1), float(l2), float(l3)))
    return candidates


def deduplicate_candidates(candidates: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    unique = {}
    for c in candidates:
        w = normalize_weights(np.array(c, dtype=np.float64))
        key = tuple(np.round(w, 6))
        if key not in unique:
            unique[key] = tuple(float(x) for x in w)
    return list(unique.values())


def load_dataset_for_experiment(dataset_name: str, task_type: str, data_rate: float):
    if dataset_name == "IDF_OilTemp":
        data = load_original_data(dataset_name, "clean", rate=data_rate)
        label = None
    else:
        data, label = load_single_dataset(task_type, dataset_name, rate=data_rate)

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)

    return data, label


def build_dataset_context(
    dataset_name: str,
    task_type: str,
    data_rate: float,
    error_ratio: float,
    error_types: List[str],
    miner_type: bool,
    seed: int,
):
    data, label = load_dataset_for_experiment(dataset_name, task_type, data_rate)

    dm = DataManager(data, label=label, task_type=task_type, seed=seed)
    n_timestamps = int(data.shape[1]) if data.ndim == 3 else int(data.shape[0])

    if n_timestamps < 40 or error_ratio <= 0:
        dm = DataManager(
            data,
            label=label,
            observed_data=data.copy(),
            task_type=task_type,
            seed=seed,
        )
    else:
        try:
            dm.inject_errors(error_ratio, error_types, covered_attrs=range(data.shape[-1]))
        except ValueError:
            dm = DataManager(
                data,
                label=label,
                observed_data=data.copy(),
                task_type=task_type,
                seed=seed,
            )

    detector = Detector(dm, miner_type=miner_type)
    _, _, _, _, constraints, data_to_repair = detector.detect_all()

    return {
        "dataset": dataset_name,
        "label": label,
        "dirty_data": data_to_repair,
        "clean_data": dm.clean_data_raw.copy(),
        "detector": detector,
        "constraints": constraints,
    }


def summarize_detail_metrics(details_list: List[Dict]) -> Dict[str, float]:
    if not details_list:
        return {
            "final_test_perf": np.nan,
            "perf_improvement": np.nan,
            "total_time_cost": np.nan,
            "best_k": np.nan,
        }

    return {
        "final_test_perf": float(np.mean([d["final_test_perf"] for d in details_list])),
        "perf_improvement": float(np.mean([d["perf_improvement"] for d in details_list])),
        "total_time_cost": float(np.mean([d["total_time_cost"] for d in details_list])),
        "best_k": float(np.mean([d["best_k"] for d in details_list])),
    }


def summarize_metric_dicts(metrics_list: List[Dict]) -> Dict[str, float]:
    if not metrics_list:
        return {
            "f1_score": np.nan,
            "rra": np.nan,
            "mse": np.nan,
            "time_cost": np.nan,
        }

    return {
        "f1_score": float(np.mean([m.get("f1_score", np.nan) for m in metrics_list])),
        "rra": float(np.mean([m.get("rra", np.nan) for m in metrics_list])),
        "mse": float(np.mean([m.get("mse", np.nan) for m in metrics_list])),
        "time_cost": float(np.mean([m.get("time_cost", np.nan) for m in metrics_list])),
    }


def evaluate_candidate(
    lambdas: Tuple[float, float, float],
    dataset_contexts: Dict[str, Dict],
    task_type: str,
    max_steps: int,
    n_episodes: int,
    miner_type: bool,
    repeats: int,
) -> Dict:
    l1, l2, l3 = normalize_weights(np.array(lambdas, dtype=np.float64))

    if len(dataset_contexts) != 1:
        raise ValueError("当前模式仅支持一次评估一个数据集，请只传入一个 dataset")

    dataset_name, ctx = next(iter(dataset_contexts.items()))

    run_details = []
    run_eval_metrics = []

    for _ in range(repeats):
        set_lambda_weights(l1, l2, l3, normalize=True)
        repaired_data, time_cost, _, details = hrl_main(
            dirty_data=ctx["dirty_data"].copy(),
            label=ctx["label"],
            task_type=task_type,
            detector=ctx["detector"],
            constraints=ctx["constraints"],
            concentrated=miner_type,
            max_steps=max_steps,
            n_episodes=n_episodes,
            return_details=True,
        )
        run_details.append(details)

        eval_results = evaluate_cleaning_effectiveness(
            repaired_data,
            ctx["clean_data"],
            ctx["dirty_data"],
            time_cost,
        )
        print(
            "[关键指标] "
            f"dataset={dataset_name}, "
            f"repeat={len(run_eval_metrics) + 1}/{repeats}, "
            f"mse={eval_results.get('mse', np.nan):.6f}, "
            f"mnad={eval_results.get('mnad', np.nan):.6f}, "
            f"mae_error={eval_results.get('mae_error', np.nan):.6f}, "
            f"rra={eval_results.get('rra', np.nan):.6f}, "
            f"f1={eval_results.get('f1_score', np.nan):.6f}"
        )
        run_eval_metrics.append(eval_results)

    detail_metrics = summarize_detail_metrics(run_details)
    eval_metrics = summarize_metric_dicts(run_eval_metrics)

    print(
        f"[性能提升] dataset={dataset_name}, "
        f"perf_improvement={detail_metrics.get('perf_improvement', np.nan):.6f}"
    )

    perf_improvement = float(detail_metrics["perf_improvement"])
    f1_score_metric = float(eval_metrics["f1_score"])
    rra_metric = float(eval_metrics["rra"])
    mse_metric = float(eval_metrics["mse"])
    time_cost_metric = float(eval_metrics["time_cost"])

    mse_score = 1.0 / (1.0 + max(mse_metric, 0.0))
    overall_score = 0.5 * perf_improvement + 0.2 * f1_score_metric + 0.2 * rra_metric + 0.1 * mse_score

    result = {
        "lambda_1": float(l1),
        "lambda_2": float(l2),
        "lambda_3": float(l3),
        "lambda_sum": float(l1 + l2 + l3),
        "dataset": dataset_name,
        "perf_improvement": perf_improvement,
        "f1_score": f1_score_metric,
        "rra": rra_metric,
        "mse": mse_metric,
        "time_cost": time_cost_metric,
        "overall_score": overall_score,
    }

    result[f"{dataset_name}_final_test_perf"] = detail_metrics["final_test_perf"]
    result[f"{dataset_name}_best_k"] = detail_metrics["best_k"]

    return result


def run_search(args):
    datasets = args.datasets

    if len(datasets) != 1:
        raise ValueError("当前实验要求一次只跑一个数据集，请通过 --datasets 仅传入一个数据集")

    print("开始构建实验数据上下文...")
    contexts = {
        name: build_dataset_context(
            dataset_name=name,
            task_type=args.task_type,
            data_rate=args.data_rate,
            error_ratio=args.error_ratio,
            error_types=args.error_types,
            miner_type=args.miner_type,
            seed=args.seed,
        )
        for name in datasets
    }

    candidates = generate_grid_candidates(
        step=args.grid_step,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
    )
    candidates = deduplicate_candidates(candidates)

    print(f"候选参数组数量: {len(candidates)}")
    if len(candidates) == 0:
        raise RuntimeError("没有可用的 lambda 候选组合")

    all_results = []
    total = len(candidates)

    for i, lambdas in enumerate(candidates, start=1):
        print("=" * 90)
        print(f"[{i}/{total}] 评估 lambda = {tuple(round(x, 4) for x in lambdas)}")
        start = time.time()
        try:
            row = evaluate_candidate(
                lambdas=lambdas,
                dataset_contexts=contexts,
                task_type=args.task_type,
                max_steps=args.max_steps,
                n_episodes=args.n_episodes,
                miner_type=args.miner_type,
                repeats=args.repeats,
            )
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            l1, l2, l3 = normalize_weights(np.array(lambdas, dtype=np.float64))
            row = {
                "lambda_1": float(l1),
                "lambda_2": float(l2),
                "lambda_3": float(l3),
                "lambda_sum": float(l1 + l2 + l3),
                "dataset": datasets[0] if datasets else "",
                "perf_improvement": np.nan,
                "f1_score": np.nan,
                "rra": np.nan,
                "mse": np.nan,
                "time_cost": np.nan,
                "overall_score": -np.inf,
                "status": "failed",
                "error": str(exc),
            }

        row["elapsed_seconds"] = float(time.time() - start)
        all_results.append(row)

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        by=["status", "perf_improvement", "f1_score", "rra", "mse"],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    top_df = df[df["status"] == "ok"].head(args.top_k).copy()

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_path = os.path.join(args.output_dir, f"lambda_search_all_{stamp}.csv")
    top_path = os.path.join(args.output_dir, f"lambda_search_top_{stamp}.csv")

    df.to_csv(all_path, index=False)
    top_df.to_csv(top_path, index=False)

    print("\n实验完成。")
    print(f"全部结果: {all_path}")
    print(f"Top-{args.top_k} 结果: {top_path}")

    if not top_df.empty:
        print("\nTop 参数组合：")
        show_cols = [
            "lambda_1",
            "lambda_2",
            "lambda_3",
            "lambda_sum",
            "perf_improvement",
            "f1_score",
            "rra",
            "mse",
            "overall_score",
        ]
        print(top_df[show_cols].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="自动搜索 LAMBDA_1~3（和为1）的最优比重")

    parser.add_argument("--task-type", type=str, default="forecast", choices=["forecast"], help="当前实验任务类型")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1"],
        help="要评估的数据集列表，例如: --datasets ETTh1 IDF_OilTemp",
    )
    parser.add_argument("--data-rate", type=float, default=0.2, help="数据读取比例，范围(0,1]")
    parser.add_argument("--error-ratio", type=float, default=0.7, help="注入错误比例")
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=["missing", "duplicate", "single", "drift", "gaussian", "volatility", "gradual", "sudden"],
        help="注入错误类型列表",
    )

    parser.add_argument("--grid-step", type=float, default=0.1, help="lambda 网格搜索步长，默认 0.1")
    parser.add_argument("--lambda-min", type=float, default=0.1, help="每个 lambda 的最小值")
    parser.add_argument("--lambda-max", type=float, default=0.8, help="每个 lambda 的最大值")

    parser.add_argument("--repeats", type=int, default=1, help="每组参数重复运行次数")
    parser.add_argument("--max-steps", type=int, default=10, help="每轮最大清洗步数")
    parser.add_argument("--n-episodes", type=int, default=3, help="HRL 训练 episode 数")
    parser.add_argument("--miner-type", dest="miner_type", action="store_true", default=True, help="启用统一约束挖掘")
    parser.add_argument("--no-miner-type", dest="miner_type", action="store_false", help="关闭统一约束挖掘")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于数据注入和候选生成）")

    parser.add_argument("--top-k", type=int, default=5, help="输出最优组合数量")
    parser.add_argument("--output-dir", type=str, default="/home/yyy/TSC/TSClean/AutoClean/parameter", help="结果输出目录")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(args)
