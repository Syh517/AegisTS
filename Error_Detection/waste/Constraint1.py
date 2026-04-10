import numpy as np
import pandas as pd
import re


class ConstraintViolationDetector:
    """
    综合约束违反检测器
    能够检测多种类型的约束违反情况，包括：
    1. 行约束违反
    2. 列约束违反（速度、加速度、方差、振幅
    3. Order Dependency约束违反
    4. Denial Dependency约束违反
    5. Sequential Dependency约束违反
    """

    def __init__(self, constraints):
        """
        初始化约束违反检测器

        :param constraints: 从mine_all_constraints函数得到的所有约束
        """
        self.constraints = constraints
        self.violations = []
        # 注意：在初始化时，data尚未提供，因此self.cv将在detect_all_violations中初始化
        self.time_col = "timestamp"

    def detect_od_violations(self, data):
        """
        检测 Order Dependency 约束违反（用归并排序 O(n log n)）
        返回违反的行对信息
        """
        od_violations = []
        od_constraints = self.constraints.get("od_constraints", [])
        if not od_constraints:
            return od_violations

        def merge_sort_detect(values, rows, monotonicity):
            """
            用归并排序检测逆序对
            :param values: 目标列值数组
            :param rows: 原始行索引数组
            :param monotonicity: 'increasing' or 'decreasing'
            :return: violations 列表
            """

            def merge_sort(arr, idx):
                if len(arr) <= 1:
                    return arr, idx, []
                mid = len(arr) // 2
                left, l_idx, l_viol = merge_sort(arr[:mid], idx[:mid])
                right, r_idx, r_viol = merge_sort(arr[mid:], idx[mid:])
                merged, merged_idx, violations = [], [], []
                i = j = 0
                while i < len(left) and j < len(right):
                    if monotonicity == "increasing":
                        if left[i] <= right[j]:
                            merged.append(left[i])
                            merged_idx.append(l_idx[i])
                            i += 1
                        else:
                            # violation: left[i] > right[j]
                            for k in range(i, len(left)):
                                violations.append(
                                    (l_idx[k], r_idx[j], left[k], right[j])
                                )
                            merged.append(right[j])
                            merged_idx.append(r_idx[j])
                            j += 1
                    else:  # decreasing
                        if left[i] >= right[j]:
                            merged.append(left[i])
                            merged_idx.append(l_idx[i])
                            i += 1
                        else:
                            # violation: left[i] < right[j]
                            for k in range(i, len(left)):
                                violations.append(
                                    (l_idx[k], r_idx[j], left[k], right[j])
                                )
                            merged.append(right[j])
                            merged_idx.append(r_idx[j])
                            j += 1
                # 合并剩余
                while i < len(left):
                    merged.append(left[i])
                    merged_idx.append(l_idx[i])
                    i += 1
                while j < len(right):
                    merged.append(right[j])
                    merged_idx.append(r_idx[j])
                    j += 1
                return merged, merged_idx, l_viol + r_viol + violations

            _, _, violations = merge_sort(values, rows)
            return violations

        # 遍历约束
        for i, constraint in enumerate(od_constraints):
            source_col = constraint["source"]
            target_col = constraint["target"]
            monotonicity = constraint.get("monotonicity", None)
            correlation = constraint.get("correlation", None)

            # 推断方向
            if monotonicity is None:
                monotonicity = (
                    "increasing"
                    if (correlation is not None and correlation > 0)
                    else "decreasing"
                )

            # 提取数据
            df_subset = data[[source_col, target_col]].dropna().reset_index()
            if df_subset.empty:
                continue

            df_sorted = df_subset.sort_values(by=source_col).reset_index(drop=True)
            values = df_sorted[target_col].values
            rows = df_sorted["index"].values

            # 检测违例
            violations = merge_sort_detect(values, rows, monotonicity)

            for row_i, row_j, val_i, val_j in violations:
                violation_degree = abs(val_i - val_j)
                od_violations.append(
                    {
                        "type": "order_dependency",
                        "constraint_index": i,
                        "source_column": source_col,
                        "target_column": target_col,
                        "row_index_i": int(row_i),
                        "row_index_j": int(row_j),
                        "row_index": int(row_j),
                        "target_value_i": float(val_i),
                        "target_value_j": float(val_j),
                        "monotonicity": monotonicity,
                        "violation_degree": float(violation_degree),
                        'columns_involved': [target_col],
                    }
                )
                # 更新 self.cv
                # self.cv.loc[row_i, target_col] += violation_degree
                if target_col != self.time_col:
                    self.cv.loc[row_j, target_col] += violation_degree

        return od_violations

    def detect_column_violations(self, data):
        """
        检测列约束违反（速度、加速度、方差、振幅）

        :param data: 待检测的数据
        :return: 违反列表
        """
        column_violations = []

        # 获取列约束
        speed_constraints = self.constraints.get("speed_constraints", {})
        accel_constraints = self.constraints.get("accel_constraints", {})
        variance_constraints = self.constraints.get("variance_constraints", {})
        amplitude_constraints = self.constraints.get("amplitude_constraints", {})

        # 检查每个列的约束
        for col in data.columns:
            series = data[col]

            # 检查速度约束
            if col in speed_constraints:
                speed_range = speed_constraints[col]
                diffs = series.diff()
                # 检查是否有超出速度范围的点
                violation_indices = diffs[
                    (diffs < speed_range[0]) | (diffs > speed_range[1])
                ].index
                for idx in violation_indices:
                    if idx > 0:  # 第一行没有diff
                        # 计算违反程度
                        value = diffs[idx]
                        if value < speed_range[0]:
                            violation_degree = speed_range[0] - value
                        else:  # value > speed_range[1]
                            violation_degree = value - speed_range[1]

                        column_violations.append(
                            {
                                "type": "column_speed",
                                "column": col,
                                "column_index": col,  # 明确标明列索引
                                "row_index": idx,  # 明确标明行索引
                                "value": value,
                                "lower_bound": speed_range[0],
                                "upper_bound": speed_range[1],
                                "violation_degree": violation_degree,
                                'columns_involved': [col],
                            }
                        )

                        # 将违反程度累积到self.cv中的对应行列
                        self.cv.loc[idx, col] += violation_degree

            # 检查加速度约束
            if col in accel_constraints:
                accel_range = accel_constraints[col]
                accels = series.diff().diff()
                # 检查是否有超出加速度范围的点
                violation_indices = accels[
                    (accels < accel_range[0]) | (accels > accel_range[1])
                ].index
                for idx in violation_indices:
                    if idx > 1:  # 前两行没有二阶diff
                        # 计算违反程度
                        value = accels[idx]
                        if value < accel_range[0]:
                            violation_degree = accel_range[0] - value
                        else:  # value > accel_range[1]
                            violation_degree = value - accel_range[1]

                        column_violations.append(
                            {
                                "type": "column_acceleration",
                                "column": col,
                                "column_index": col,  # 明确标明列索引
                                "row_index": idx,  # 明确标明行索引
                                "value": value,
                                "lower_bound": accel_range[0],
                                "upper_bound": accel_range[1],
                                "violation_degree": violation_degree,
                                'columns_involved': [col],
                            }
                        )

                        # 将违反程度累积到self.cv中的对应行列
                        self.cv.loc[idx, col] += violation_degree

            # 检查方差约束
            if col in variance_constraints:
                variance_range = variance_constraints[col]
                window_size = min(10, len(series))  # 使用滑动窗口计算局部方差
                for i in range(len(series) - window_size + 1):
                    window_data = series[i : i + window_size]
                    local_variance = np.var(window_data)
                    if not (variance_range[0] <= local_variance <= variance_range[1]):
                        # 计算违反程度
                        if local_variance < variance_range[0]:
                            violation_degree = variance_range[0] - local_variance
                        else:  # local_variance > variance_range[1]
                            violation_degree = local_variance - variance_range[1]

                        column_violations.append(
                            {
                                "type": "column_variance",
                                "column": col,
                                "column_index": col,  # 明确标明列索引
                                "row_index": i
                                + window_size // 2,  # 窗口中心点，明确标明行索引
                                "value": local_variance,
                                "lower_bound": variance_range[0],
                                "upper_bound": variance_range[1],
                                "violation_degree": violation_degree,
                                'columns_involved': [col],
                            }
                        )

                        # 将违反程度累积到self.cv中的对应行列（整个窗口范围）
                        for window_idx in range(i, i + window_size):
                            self.cv.loc[window_idx, col] += violation_degree

            # 检查振幅约束
            if col in amplitude_constraints:
                amplitude_range = amplitude_constraints[col]
                # 检查是否有超出振幅范围的点
                violation_indices = series[
                    (series < amplitude_range[0]) | (series > amplitude_range[1])
                ].index
                for idx in violation_indices:
                    # 计算违反程度
                    value = series[idx]
                    if value < amplitude_range[0]:
                        violation_degree = amplitude_range[0] - value
                    else:  # value > amplitude_range[1]
                        violation_degree = value - amplitude_range[1]

                    column_violations.append(
                        {
                            "type": "column_amplitude",
                            "column": col,
                            "column_index": col,  # 明确标明列索引
                            "row_index": idx,  # 明确标明行索引
                            "value": value,
                            "lower_bound": amplitude_range[0],
                            "upper_bound": amplitude_range[1],
                            "violation_degree": violation_degree,
                            'columns_involved': [col],
                        }
                    )

                    # 将违反程度累积到self.cv中的对应行列
                    self.cv.loc[idx, col] += violation_degree

        return column_violations

    def detect_row_violations(self, data):
        """
        检测行约束违反

        :param data: 待检测的数据
        :return: 违反列表
        """
        row_violations = []

        # 获取行约束
        row_constraints = self.constraints.get("row_constraints", [])

        if not row_constraints:
            return row_violations

        # 检查每个行约束
        for i, constraint in enumerate(row_constraints):
            formula = constraint["formula"]
            target = constraint["target"]
            features = constraint["features"]
            coef = constraint["coef"]
            intercept = constraint["intercept"]
            lower = constraint["lower"]
            upper = constraint["upper"]

            # 对每个数据点检查约束
            for idx, row in data.iterrows():
                # 计算预测值
                pred_value = intercept
                for feature in features:
                    feature_name = feature if isinstance(feature, str) else str(feature)
                    if feature_name in coef:
                        pred_value += coef[feature_name] * row[feature_name]

                # 计算残差
                residual = row[target] - pred_value

                # 检查是否违反约束
                if not (lower <= residual <= upper):
                    # 计算违反程度
                    if residual < lower:
                        violation_degree = lower - residual
                    else:  # residual > upper
                        violation_degree = residual - upper

                    row_violations.append(
                        {
                            "type": "row_constraint",
                            "constraint_index": i,
                            "target_column": target,  # 明确标明目标列索引
                            "feature_columns": features,  # 明确标明特征列索引
                            "row_index": idx,  # 明确标明行索引
                            "actual_value": row[target],
                            "predicted_value": pred_value,
                            "residual": residual,
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "formula": formula,
                            "violation_degree": violation_degree,
                            "columns_involved": [target] + features,
                        }
                    )

                    # 将违反程度累积到self.cv中的整行
                    for col in data.columns:
                        if col != self.time_col:
                            self.cv.loc[idx, col] += violation_degree

        return row_violations

    def detect_sd_violations(self, data):
        """
        检测Sequential Dependency约束违反（支持条件区间 ranges）

        :param data: 待检测的数据（DataFrame）
        :return: 违反列表
        """
        sd_violations = []

        # 获取SD约束
        sd_constraints = self.constraints.get("sd_constraints", [])

        if not sd_constraints:
            return sd_violations

        # 检查每个SD约束
        for i, constraint in enumerate(sd_constraints):
            M = constraint["M"][0]  # 排序列名
            N = constraint["N"]  # 数值列名
            g = constraint["g"]  # 允许变化范围 (g_min, g_max)
            ranges = constraint[
                "ranges"
            ]  # 条件区间列表: [(low1, high1), (low2, high2), ...]

            # 类型检查
            if not isinstance(ranges, (list, tuple)) or len(ranges) == 0:
                continue  # 无效 ranges，跳过

            # 提取 M 的上下界用于判断
            def is_in_ranges(value, ranges_list):
                """判断 value 是否在任意一个 range 中"""
                for r in ranges_list:
                    if len(r) != 2:
                        continue
                    low, high = r
                    if low <= value <= high:
                        return True
                return False

            # 按 M 排序，并保留原始索引以便定位
            df_sorted = data.sort_values(by=M).reset_index()

            # 计算 N 的相邻差值（从第1行开始）
            delta = df_sorted[N].diff()

            # 遍历每一行（从第1行开始，因为第0行无前驱）
            for idx in range(1, len(df_sorted)):
                d = delta.iloc[idx]
                m_val = df_sorted[M].iloc[idx]

                # 判断当前 M 值是否在需要检查的 ranges 区间内
                if is_in_ranges(m_val, ranges):
                    # 只有在 ranges 内才检查约束
                    if not (g[0] <= d <= g[1]):
                        violation_degree = (g[0] - d) if d < g[0] else (d - g[1])

                        sd_violations.append(
                            {
                                "type": "sequential_dependency",
                                "constraint_index": i,
                                "sorting_column": M,  # 明确标明排序列索引
                                "value_column": N,  # 明确标明值列索引
                                "row_index_sorted": idx,  # 排序后行号
                                "row_index": df_sorted["index"].iloc[idx],  # 原始行号
                                "M_value": m_val,
                                "N_value": df_sorted[N].iloc[idx],
                                "delta": float(d),
                                "g_min": g[0],
                                "g_max": g[1],
                                "violation_degree": violation_degree,
                                "range_condition": ranges,  # 记录条件区间，便于调试,
                                "columns_involved": N,
                            }
                        )

                        # 将违反程度累积到self.cv中的对应行列
                        original_row_idx = df_sorted["index"].iloc[idx]
                        # self.cv.loc[original_row_idx, M] += violation_degree
                        if N != self.time_col:
                            self.cv.loc[original_row_idx, N] += violation_degree

        return sd_violations

    def detect_dd_violations(self, data):
        """
        检测 Denial Dependency 约束违反，并将违反程度累加到 self.cv 中

        :param data: 待检测的 DataFrame
        :return: dd_violations 列表，每条记录是 dict
        """
        dd_violations = []
        dd_constraints = self.constraints.get("dd_constraints", [])
        if not dd_constraints:
            return dd_violations

        N = len(data)
        for i, constraint in enumerate(dd_constraints):
            dd_formula = constraint.get("dd")
            support = constraint.get("support")
            confidence = constraint.get("confidence")
            if not dd_formula or not isinstance(dd_formula, str):
                continue

            # 去掉 ¬ 和外层括号
            inner = dd_formula.strip()
            if inner.startswith("¬"):
                inner = inner[1:].strip()
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1].strip()

            clauses = re.split(r"\s+and\s+", inner)
            if len(clauses) < 2:
                try:
                    violated_idx = data.query(inner).index.tolist()
                except Exception:
                    continue
                for idx in violated_idx:
                    columns_involved = set()
                    for cl in clauses:
                        m = re.match(r"\s*([A-Za-z_]\w*)", cl.strip())
                        if m:
                            columns_involved.add(m.group(1))
                    violation_degree = 1.0
                    # 累加到 self.cv
                    for col in columns_involved:
                        if col in self.cv.columns and col != self.time_col:
                            self.cv.loc[idx, col] += violation_degree
                    # 添加到返回结果
                    dd_violations.append(
                        {
                            "type": "denial_dependency",
                            "constraint_index": i,
                            "row_index": idx,
                            "formula": dd_formula,
                            "support": support,
                            "confidence": confidence,
                            "violation_degree": violation_degree,
                            "columns_involved": list(columns_involved),
                        }
                    )
                continue

            # 按列分组
            groups = {}
            for cl in clauses:
                cl_strip = cl.strip()
                m = re.match(r"\s*([A-Za-z_]\w*)", cl_strip)
                col = m.group(1) if m else "__misc__"
                groups.setdefault(col, []).append(cl_strip)

            if len(groups) == 2:
                p_exprs = [" and ".join(v) for v in groups.values()]
                p1_expr, p2_expr = p_exprs[0], p_exprs[1]
                col_list = list(groups.keys())
            else:
                mid = max(1, len(clauses) // 2)
                p1_expr = " and ".join(clauses[:mid])
                p2_expr = " and ".join(clauses[mid:])
                col_list = list(groups.keys())

            # 向量化计算 mask1 / mask2
            try:
                mask1 = data.eval(p1_expr)
                mask2 = data.eval(p2_expr)
                violated_mask = mask1 & mask2
                related_mask = mask1 | mask2
            except Exception:
                try:
                    violated_idx = data.query(inner).index.tolist()
                except Exception:
                    continue
                for idx in violated_idx:
                    columns_involved = set()
                    for cl in clauses:
                        m = re.match(r"\s*([A-Za-z_]\w*)", cl.strip())
                        if m:
                            columns_involved.add(m.group(1))
                    violation_degree = 1.0
                    for col in columns_involved:
                        if col in self.cv.columns and col != self.time_col:
                            self.cv.loc[idx, col] += violation_degree
                    dd_violations.append(
                        {
                            "type": "denial_dependency",
                            "constraint_index": i,
                            "row_index": idx,
                            "formula": dd_formula,
                            "support": support,
                            "confidence": confidence,
                            "violation_degree": violation_degree,
                            "columns_involved": list(columns_involved),
                        }
                    )
                continue

            violated_count = int(violated_mask.sum())
            related_count = int(related_mask.sum())
            if violated_count == 0:
                continue
            violation_degree = (
                (violated_count / related_count) if related_count > 0 else 1.0
            )

            # 向量化计算哪些列参与违反
            col_bool_map = pd.DataFrame(False, index=data.index, columns=col_list)
            for col, clist in groups.items():
                col_mask = pd.Series(False, index=data.index)
                for cl in clist:
                    expr = cl.replace(col, f"data['{col}']")
                    try:
                        col_mask |= data.eval(expr)
                    except Exception:
                        continue
                col_bool_map[col] = col_mask

            violated_rows = data.index[violated_mask]
            for idx in violated_rows:
                cols_involved = col_bool_map.loc[idx]
                involved_list = cols_involved[cols_involved].index.tolist()
                # 累加到 self.cv
                for col in involved_list:
                    if col in self.cv.columns and col != self.time_col:
                        self.cv.loc[idx, col] += violation_degree
                dd_violations.append(
                    {
                        "type": "denial_dependency",
                        "constraint_index": i,
                        "row_index": idx,
                        "formula": dd_formula,
                        "support": support,
                        "confidence": confidence,
                        "violation_degree": float(violation_degree),
                        "columns_involved": involved_list,
                    }
                )

        return dd_violations

    def detect_all_violations(self, data):
        """
        检测所有类型的约束违反

        :param data: 待检测的数据
        :return: 所有违反的列表
        """
        # 初始化self.cv
        self.cv = pd.DataFrame(
            0.0, index=data.index, columns=data.columns, dtype=float  # 数值 0.0
        )

        all_violations = []

        # 检测OD约束违反
        od_violations = self.detect_od_violations(data)
        all_violations.extend(od_violations)

        # 检测列约束违反
        column_violations = self.detect_column_violations(data)
        all_violations.extend(column_violations)

        # 检测行约束违反
        row_violations = self.detect_row_violations(data)
        all_violations.extend(row_violations)


        # 检测SD约束违反
        sd_violations = self.detect_sd_violations(data)
        all_violations.extend(sd_violations)

        # 检测DD约束违反
        dd_violations = self.detect_dd_violations(data)
        all_violations.extend(dd_violations)

        self.violations = all_violations
        return all_violations



def get_violation_summary(violations):
    """
    获取违反情况的汇总统计

    :return: 违反情况的统计信息
    """
    if not violations:
        return "没有发现约束违反"

    od_count = sum(1 for v in violations if v["type"] == "order_dependency")
    row_count = sum(1 for v in violations if v["type"] == "row_constraint")
    col_speed_count = sum(1 for v in violations if v["type"] == "column_speed")
    col_accel_count = sum(
        1 for v in violations if v["type"] == "column_acceleration"
    )
    col_var_count = sum(
        1 for v in violations if v["type"] == "column_variance"
    )
    col_amp_count = sum(
        1 for v in violations if v["type"] == "column_amplitude"
    )
    sd_count = sum(
        1 for v in violations if v["type"] == "sequential_dependency"
    )
    dd_count = sum(1 for v in violations if v["type"] == "denial_dependency")

    summary = {
        "total_violations": len(violations),
        "order_dependency_violations": od_count,
        "row_constraint_violations": row_count,
        "column_speed_violations": col_speed_count,
        "column_acceleration_violations": col_accel_count,
        "column_variance_violations": col_var_count,
        "column_amplitude_violations": col_amp_count,
        "sequential_dependency_violations": sd_count,
        "denial_dependency_violations": dd_count,
    }

    return summary


def constraint_violations_report(violations):
    # 输出结果
    print("约束违反检测结果:")
    print(f"总共发现 {len(violations)} 个违反")


    summary = get_violation_summary(violations)
    print("\n违反统计:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


    for i, violation in enumerate(violations[:]):
        print(f"  {i+1}. 类型: {violation['type']}")
        if "row_index" in violation:
            print(f"     行索引: {violation['row_index']}")
        if "columns_involved" in violation:
            print(f"     列名: {violation['columns_involved']}")
        print(f"     违反程度: {violation.get('violation_degree', 'N/A')}")
        print()


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    data = {
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    }
    data = pd.DataFrame(data)

    # np.random.seed(42)
    # data = pd.DataFrame({
    #     "timestamp": range(100),
    #     "temperature": np.random.normal(20, 5, 100),
    #     "humidity": np.random.uniform(30, 70, 100)
    # })

    # 模拟约束（实际应该通过mine_all_constraints获取）
    from Error_Detection.miner_1 import mine_all_constraints, constraint_report

    attr_num = data.shape[1]
    constraints = mine_all_constraints(
        data,
        degree=3,
        attr_num=attr_num,
        window=20,
        strategy="auto",
        min_support=0.1,
        min_conf=0.9,
    )
    # constraint_report(constraints)

    # 创建违反检测器
    detector = ConstraintViolationDetector(constraints)

    # 检测违反
    violations = detector.detect_all_violations(data)

    constraint_violations_report(violations)

    print(detector.cv)

