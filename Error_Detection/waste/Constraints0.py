import numpy as np
import pandas as pd
import re

from Error_Detection.miner0 import mine_all_constraints, constraint_report
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ConstraintViolationDetector:
    """
    综合约束违反检测器
    能够检测多种类型的约束违反情况，包括：
    1. 行约束违反
    2. 列约束违反（速度、加速度、方差、振幅）
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
        self.time_col = "timestamp"  # timestamp列保护

    def detect_od_violations(self, data):
        """
        检测 Order Dependency 约束违反（归并排序 O(n log n)）
        返回违反的行对信息
        """
        od_violations = []
        od_constraints = self.constraints.get("od_constraints", [])
        if not od_constraints:
            return od_violations

        def merge_sort_detect(values, rows, monotonicity):
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
                            for k in range(i, len(left)):
                                violations.append(
                                    (l_idx[k], r_idx[j], left[k], right[j])
                                )
                            merged.append(right[j])
                            merged_idx.append(r_idx[j])
                            j += 1
                while i < len(left):
                    merged.append(left[i])
                    merged_idx.append(l_idx[i])
                    i += 1
                while j < len(right):
                    merged.append(right[j])
                    merged_idx.append(r_idx[j])
                    j += 1
                return merged, merged_idx, violations

            _, _, violations = merge_sort(values, rows)
            return violations

        for i, constraint in enumerate(od_constraints):
            source_col = constraint["source"]
            target_col = constraint["target"]
            monotonicity = constraint.get("monotonicity")
            correlation = constraint.get("correlation")

            if monotonicity is None:
                monotonicity = (
                    "increasing"
                    if (correlation is not None and correlation > 0)
                    else "decreasing"
                )

            df_subset = data[[source_col, target_col]].dropna().reset_index()
            if df_subset.empty:
                continue

            df_sorted = df_subset.sort_values(by=source_col).reset_index(drop=True)
            values = df_sorted[target_col].values
            rows = df_sorted["index"].values

            violations = merge_sort_detect(values, rows, monotonicity)

            # 使用集合跟踪已记录的违反行，避免重复记录同一行
            recorded_rows = set()

            for row_i, row_j, val_i, val_j in violations:
                # 只有当行尚未记录时才添加违反记录
                if row_j not in recorded_rows:
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
                            "columns_involved": [target_col],
                        }
                    )
                    if target_col != self.time_col:
                        self.cv.loc[row_j, target_col] += violation_degree
                    recorded_rows.add(row_j)
        return od_violations

    def detect_column_violations(self, data):
        column_violations = []
        speed_constraints = self.constraints.get("speed_constraints", {})
        accel_constraints = self.constraints.get("accel_constraints", {})
        variance_constraints = self.constraints.get("variance_constraints", {})
        amplitude_constraints = self.constraints.get("amplitude_constraints", {})

        for col in data.columns:
            series = data[col]

            if col in speed_constraints:
                speed_range = speed_constraints[col]
                diffs = series.diff()
                violation_indices = diffs[
                    (diffs < speed_range[0]) | (diffs > speed_range[1])
                ].index
                for idx in violation_indices:
                    if idx > 0:
                        value = diffs[idx]
                        violation_degree = (
                            (speed_range[0] - value)
                            if value < speed_range[0]
                            else (value - speed_range[1])
                        )
                        column_violations.append(
                            {
                                "type": "column_speed",
                                "column": col,
                                "column_index": col,
                                "row_index": idx,
                                "value": value,
                                "lower_bound": speed_range[0],
                                "upper_bound": speed_range[1],
                                "violation_degree": violation_degree,
                                "columns_involved": [col],
                            }
                        )
                        if col != self.time_col:
                            self.cv.loc[idx, col] += violation_degree

            if col in accel_constraints:
                accel_range = accel_constraints[col]
                accels = series.diff().diff()
                violation_indices = accels[
                    (accels < accel_range[0]) | (accels > accel_range[1])
                ].index
                for idx in violation_indices:
                    if idx > 1:
                        value = accels[idx]
                        violation_degree = (
                            (accel_range[0] - value)
                            if value < accel_range[0]
                            else (value - accel_range[1])
                        )
                        column_violations.append(
                            {
                                "type": "column_acceleration",
                                "column": col,
                                "column_index": col,
                                "row_index": idx,
                                "value": value,
                                "lower_bound": accel_range[0],
                                "upper_bound": accel_range[1],
                                "violation_degree": violation_degree,
                                "columns_involved": [col],
                            }
                        )
                        if col != self.time_col:
                            self.cv.loc[idx, col] += violation_degree

            if col in variance_constraints:
                variance_range = variance_constraints[col]
                window_size = min(10, len(series))
                for i in range(len(series) - window_size + 1):
                    window_data = series[i : i + window_size]
                    local_variance = np.var(window_data)
                    if not (variance_range[0] <= local_variance <= variance_range[1]):
                        violation_degree = (
                            (variance_range[0] - local_variance)
                            if local_variance < variance_range[0]
                            else (local_variance - variance_range[1])
                        )
                        column_violations.append(
                            {
                                "type": "column_variance",
                                "column": col,
                                "column_index": col,
                                "row_index": i + window_size // 2,
                                "value": local_variance,
                                "lower_bound": variance_range[0],
                                "upper_bound": variance_range[1],
                                "violation_degree": violation_degree,
                                "columns_involved": [col],
                            }
                        )
                        if col != self.time_col:
                            for window_idx in range(i, i + window_size):
                                self.cv.loc[window_idx, col] += violation_degree

            if col in amplitude_constraints:
                amplitude_range = amplitude_constraints[col]
                violation_indices = series[
                    (series < amplitude_range[0]) | (series > amplitude_range[1])
                ].index
                for idx in violation_indices:
                    value = series[idx]
                    violation_degree = (
                        (amplitude_range[0] - value)
                        if value < amplitude_range[0]
                        else (value - amplitude_range[1])
                    )
                    column_violations.append(
                        {
                            "type": "column_amplitude",
                            "column": col,
                            "column_index": col,
                            "row_index": idx,
                            "value": value,
                            "lower_bound": amplitude_range[0],
                            "upper_bound": amplitude_range[1],
                            "violation_degree": violation_degree,
                            "columns_involved": [col],
                        }
                    )
                    if col != self.time_col:
                        self.cv.loc[idx, col] += violation_degree

        return column_violations

    # def detect_row_violations(self, data):
    #     row_violations = []
    #     row_constraints = self.constraints.get("row_constraints", [])
    #     if not row_constraints:
    #         return row_violations

    #     for i, constraint in enumerate(row_constraints):
    #         formula = constraint["formula"]
    #         target = constraint["target"]
    #         features = constraint["features"]
    #         coef = constraint["coef"]
    #         intercept = constraint["intercept"]
    #         lower = constraint["lower"]
    #         upper = constraint["upper"]

    #         for idx, row in data.iterrows():
    #             pred_value = intercept
    #             for feature in features:
    #                 feature_name = feature if isinstance(feature, str) else str(feature)
    #                 if feature_name in coef:
    #                     pred_value += coef[feature_name] * row[feature_name]

    #             residual = row[target] - pred_value
    #             if not (lower <= residual <= upper):
    #                 violation_degree = (
    #                     (lower - residual) if residual < lower else (residual - upper)
    #                 )
    #                 row_violations.append(
    #                     {
    #                         "type": "row_constraint",
    #                         "constraint_index": i,
    #                         "target_column": target,
    #                         "feature_columns": features,
    #                         "row_index": idx,
    #                         "actual_value": row[target],
    #                         "predicted_value": pred_value,
    #                         "residual": residual,
    #                         "lower_bound": lower,
    #                         "upper_bound": upper,
    #                         "formula": formula,
    #                         "violation_degree": violation_degree,
    #                         "columns_involved": [target] + features,
    #                     }
    #                 )
    #                 for col in data.columns:
    #                     if col != self.time_col:
    #                         self.cv.loc[idx, col] += violation_degree

    #     return row_violations


    def detect_row_violations(self, data):
        """
        检测数据行是否违反“多变量关系型”行约束
        约束形式：target ≈ f(features)，残差 ∈ [lower, upper]
        """
        row_violations = []
        row_constraints = self.constraints.get("row_constraints", [])
        if not row_constraints:
            return row_violations

        for i, constraint in enumerate(row_constraints):
            formula = constraint["formula"]
            target = constraint["target"]
            features = constraint["features"]
            coef = constraint["coef"]           # 系数字典，如 {"X": 0.5, "X Y": 0.1}
            intercept = constraint["intercept"]
            lower = constraint["lower"]         # 残差下界
            upper = constraint["upper"]         # 残差上界

            # 检查所需列是否存在
            required_cols = set(features)
            if not required_cols.issubset(data.columns):
                continue  # 跳过缺失列的约束

            for idx, row in data.iterrows():
                # 构造多项式特征（必须与训练时一致）
                X_row = np.array([row[f] for f in features]).reshape(1, -1)

                # 使用 PolynomialFeatures 构建交叉项（degree=2）
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly.fit(X_row)  # 仅拟合结构
                X_poly_row = poly.transform(X_row)
                feature_names = poly.get_feature_names_out(features)

                # 计算预测值：pred = intercept + sum(coef * x_poly)
                pred_value = intercept
                for j, name in enumerate(feature_names):
                    if name in coef:
                        pred_value += coef[name] * X_poly_row[0, j]

                actual_value = row[target]
                residual = actual_value - pred_value

                # 判断是否违反残差容忍区间
                if residual < lower:
                    violation_degree = lower - residual  # 负向违反
                elif residual > upper:
                    violation_degree = residual - upper  # 正向违反
                else:
                    continue  # 无违反

                # 记录违规
                row_violations.append({
                    "type": "row_constraint",
                    "constraint_index": i,
                    "target_column": target,
                    "feature_columns": features,
                    "row_index": idx,
                    "actual_value": actual_value,
                    "predicted_value": pred_value,
                    "residual": residual,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "formula": formula,
                    "violation_degree": violation_degree,
                    "columns_involved": [target] + features,
                })

                # 累加违反度到 cv（可选：加权或平方）
                for col in [target] + features:
                    if col != self.time_col:
                        self.cv.loc[idx, col] += violation_degree

        return row_violations

    def detect_sd_violations(self, data):
        sd_violations = []
        sd_constraints = self.constraints.get("sd_constraints", [])
        if not sd_constraints:
            return sd_violations

        for i, constraint in enumerate(sd_constraints):
            M = constraint["M"][0]
            N = constraint["N"]
            g = constraint["g"]
            ranges = constraint["ranges"]

            def is_in_ranges(value, ranges_list):
                for r in ranges_list:
                    if len(r) != 2:
                        continue
                    low, high = r
                    if low <= value <= high:
                        return True
                return False

            df_sorted = data.sort_values(by=M).reset_index()
            delta = df_sorted[N].diff()

            for idx in range(1, len(df_sorted)):
                d = delta.iloc[idx]
                m_val = df_sorted[M].iloc[idx]

                if is_in_ranges(m_val, ranges):
                    if not (g[0] <= d <= g[1]):
                        violation_degree = (g[0] - d) if d < g[0] else (d - g[1])
                        sd_violations.append(
                            {
                                "type": "sequential_dependency",
                                "constraint_index": i,
                                "sorting_column": M,
                                "value_column": N,
                                "row_index_sorted": idx,
                                "row_index": df_sorted["index"].iloc[idx],
                                "M_value": m_val,
                                "N_value": df_sorted[N].iloc[idx],
                                "delta": float(d),
                                "g_min": g[0],
                                "g_max": g[1],
                                "violation_degree": violation_degree,
                                "range_condition": ranges,
                                "columns_involved": N,
                            }
                        )
                        original_row_idx = df_sorted["index"].iloc[idx]
                        if N != self.time_col:
                            self.cv.loc[original_row_idx, N] += violation_degree
        return sd_violations

    def detect_dd_violations(self, data):
        dd_violations = []
        dd_constraints = self.constraints.get("dd_constraints", [])
        if not dd_constraints:
            return dd_violations

        for i, constraint in enumerate(dd_constraints):
            dd_formula = constraint.get("dd")
            support = constraint.get("support")
            confidence = constraint.get("confidence")
            if not dd_formula or not isinstance(dd_formula, str):
                continue

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

            try:
                mask1 = data.eval(p1_expr)
                mask2 = data.eval(p2_expr)
                violated_mask = mask1 & mask2
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
                for col in involved_list:
                    if col in self.cv.columns and col != self.time_col:
                        self.cv.loc[idx, col] += violated_mask.sum() / max(
                            1, (mask1 | mask2).sum()
                        )
                dd_violations.append(
                    {
                        "type": "denial_dependency",
                        "constraint_index": i,
                        "row_index": idx,
                        "formula": dd_formula,
                        "support": support,
                        "confidence": confidence,
                        "violation_degree": float(
                            violated_mask.sum() / max(1, (mask1 | mask2).sum())
                        ),
                        "columns_involved": involved_list,
                    }
                )
        return dd_violations

    # def detect_td_violations(self, data, window_size=10):
    #     """
    #     检测趋势依赖约束违反情况，并将违反度累加到 cv 对应行列上

    #     :param df: 原始数据 DataFrame
    #     :param dependencies: 趋势依赖约束列表，每个 dict 包含 M, N, trend_type, trend_confidence
    #     :param window_size: 滑动窗口均值平滑
    #     :return: violations 列表，每个元素为 dict 包含行列信息和违反度
    #     """

    
    #     td_violations = []
    #     td_constraints = self.constraints.get("td_constraints", [])
    #     if not td_constraints:
    #         return td_violations
        
    #     df = data.copy()
    #     td_violations = []

    #     # 平滑函数
    #     def window_smooth(series):
    #         return series.rolling(window=window_size, min_periods=1).mean().values

    #     for i, dep in enumerate(td_constraints):
    #         M_col = dep['M']
    #         N_col = dep['N']
    #         trend_type = dep['trend_type']
    #         confidence = dep.get('trend_confidence', 1.0)

    #         if M_col == self.time_col or N_col == self.time_col:
    #             continue

    #         M_vals = window_smooth(df[M_col])
    #         N_vals = window_smooth(df[N_col])

    #         delta_M = np.diff(M_vals)
    #         delta_N = np.diff(N_vals)

    #         violated_idx = []
    #         violation_degrees = []

    #         for idx in range(len(delta_M)):
    #             if delta_M[idx] == 0 or delta_N[idx] == 0:
    #                 continue
    #             if trend_type == 'positive' and np.sign(delta_M[idx]) != np.sign(delta_N[idx]):
    #                 violated_idx.append(idx + 1)
    #                 violation_degrees.append(abs(delta_M[idx] - delta_N[idx]))
    #             elif trend_type == 'negative' and np.sign(delta_M[idx]) == np.sign(delta_N[idx]):
    #                 violated_idx.append(idx + 1)
    #                 violation_degrees.append(abs(delta_M[idx] + delta_N[idx]))

    #         for idx, degree in zip(violated_idx, violation_degrees):
    #             columns_involved = [M_col, N_col]
    #             for col in columns_involved:
    #                 if col in self.cv.columns and col != self.time_col:
    #                     self.cv.loc[idx, col] += degree

    #             td_violations.append({
    #                 'type': 'trend_dependency',
    #                 'constraint_index': i,
    #                 'row_index': idx,
    #                 'M': M_col,
    #                 'N': N_col,
    #                 'trend_type': trend_type,
    #                 'trend_confidence': confidence,
    #                 'violation_degree': float(degree),
    #                 'columns_involved': columns_involved
    #             })

    #     return td_violations

    def detect_all_violations(self, data):
        self.cv = pd.DataFrame(0.0, index=data.index, columns=data.columns, dtype=float)
        all_violations = []
        print("1")
        all_violations.extend(self.detect_column_violations(data))
        print("2")
        all_violations.extend(self.detect_row_violations(data))
        print("3")
        all_violations.extend(self.detect_od_violations(data))
        print("4")
        all_violations.extend(self.detect_dd_violations(data))
        # print("5")        
        # all_violations.extend(self.detect_td_violations(data))

        self.violations = all_violations
        return all_violations

    def calculate_violation_rates(self, data):
        """
        计算各种约束类型的违反率
        
        :param data: 原始数据DataFrame
        :return: 包含各种约束违反率的字典
        """
        if not hasattr(self, 'violations') or not self.violations:
            raise ValueError("请先运行detect_all_violations方法检测违反情况")
        
        # 总行数
        total_rows = len(data)
        
        # 初始化违反率统计
        violation_rates = {}
        
        # 按类型分组统计违反次数，使用集合去重以避免重复计算同一行
        violation_types = {}
        column_violations = {}  # 专门用于列约束的统计
        od_violations_detail = {}  # 专门用于order dependency约束的统计
        # td_violations_detail = {}  # 专门用于trend dependency约束的统计
        
        # 使用集合存储已违反的行索引，避免重复计数
        violation_rows_by_type = {}
        
        for violation in self.violations:
            v_type = violation['type']
            
            # 初始化集合
            if v_type not in violation_rows_by_type:
                violation_rows_by_type[v_type] = set()
            
            # 添加行索引到集合中
            if 'row_index' in violation:
                violation_rows_by_type[v_type].add(violation['row_index'])
            elif 'row_index_j' in violation:  # 特别处理OD约束
                violation_rows_by_type[v_type].add(violation['row_index_j'])
            
            # 对于列约束，额外统计每列的违反次数
            if v_type.startswith('column_'):
                column = violation.get('column', violation.get('column_index'))
                if column:
                    if v_type not in column_violations:
                        column_violations[v_type] = {}
                    if column not in column_violations[v_type]:
                        column_violations[v_type][column] = set()
                    if 'row_index' in violation:
                        column_violations[v_type][column].add(violation['row_index'])
            
            # 对于order dependency约束，统计涉及的变量对
            if v_type == "order_dependency":
                source_col = violation.get('source_column')
                target_col = violation.get('target_column')
                if source_col and target_col:
                    od_pair = f"{source_col} → {target_col}"
                    if v_type not in od_violations_detail:
                        od_violations_detail[v_type] = {}
                    if od_pair not in od_violations_detail[v_type]:
                        od_violations_detail[v_type][od_pair] = set()
                    # OD约束使用row_index_j作为违反行
                    if 'row_index_j' in violation:
                        od_violations_detail[v_type][od_pair].add(violation['row_index_j'])
            
            # # 对于trend dependency约束，统计涉及的变量对
            # if v_type == "trend_dependency":
            #     source_col = violation.get('M')
            #     target_col = violation.get('N')
            #     if source_col and target_col:
            #         td_pair = f"{source_col} → {target_col}"
            #         if v_type not in td_violations_detail:
            #             td_violations_detail[v_type] = {}
            #         if td_pair not in td_violations_detail[v_type]:
            #             td_violations_detail[v_type][td_pair] = set()
            #         if 'row_index' in violation:
            #             td_violations_detail[v_type][td_pair].add(violation['row_index'])
        
        # 计算每种约束类型的违反行数
        for v_type, rows_set in violation_rows_by_type.items():
            violation_types[v_type] = len(rows_set)
        
        # 计算每种约束类型的违反率
        for v_type, count in violation_types.items():
            violation_rates[v_type] = {
                'count': count,
                'rate': count / total_rows
            }
        
        # 特别计算每列的违反率（针对列约束）
        for v_type, columns in column_violations.items():
            violation_rates[v_type]['columns'] = {}
            for column, rows_set in columns.items():
                count = len(rows_set)
                violation_rates[v_type]['columns'][column] = {
                    'count': count,
                    'rate': count / total_rows
                }
        
        # 特别计算每个变量对的违反率（针对order dependency约束）
        for v_type, pairs in od_violations_detail.items():
            violation_rates[v_type]['pairs_detail'] = {}
            for pair, rows_set in pairs.items():
                count = len(rows_set)
                violation_rates[v_type]['pairs_detail'][pair] = {
                    'count': count,
                    'rate': count / total_rows
                }
        
        # # 特别计算每个变量对的违反率（针对trend dependency约束）
        # for v_type, pairs in td_violations_detail.items():
        #     violation_rates[v_type]['pairs_detail'] = {}
        #     for pair, rows_set in pairs.items():
        #         count = len(rows_set)
        #         violation_rates[v_type]['pairs_detail'][pair] = {
        #             'count': count,
        #             'rate': count / total_rows
        #         }
        
        # 计算总体违反率
        # 合并所有类型的违反行索引，避免重复计数
        all_violation_rows = set()
        for rows_set in violation_rows_by_type.values():
            all_violation_rows.update(rows_set)
        
        total_violations = len(all_violation_rows)
        violation_rates['total'] = {
            'count': total_violations,
            'rate': total_violations / total_rows
        }
        
        return violation_rates


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
    col_accel_count = sum(1 for v in violations if v["type"] == "column_acceleration")
    col_var_count = sum(1 for v in violations if v["type"] == "column_variance")
    col_amp_count = sum(1 for v in violations if v["type"] == "column_amplitude")
    # sd_count = sum(1 for v in violations if v["type"] == "sequential_dependency")
    dd_count = sum(1 for v in violations if v["type"] == "denial_dependency")
    # td_count = sum(1 for v in violations if v["type"] == "trend_dependency")



    summary = {
        "total_violations": len(violations),
        "order_dependency_violations": od_count,
        "row_constraint_violations": row_count,
        "column_speed_violations": col_speed_count,
        "column_acceleration_violations": col_accel_count,
        "column_variance_violations": col_var_count,
        "column_amplitude_violations": col_amp_count,
        # "sequential_dependency_violations": sd_count,
        "denial_dependency_violations": dd_count,
        # "trend_dependency_violations": td_count,
    }

    print("\n违反统计:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary


def get_violation_rates(violations, data):
    """
    计算各种约束类型的违反率
    
    :param violations: 违反列表
    :param data: 原始数据DataFrame
    :return: 包含各种约束违反率的字典
    """
    if not violations:
        return "没有发现约束违反"
    
    total_rows = len(data)
    
    # 按类型分组统计违反次数，使用集合去重以避免重复计算同一行
    violation_types = {}
    column_violations_detail = {}  # 专门用于列约束的详细统计
    od_violations_detail = {}  # 专门用于order dependency约束的统计
    # td_violations_detail = {}  # 专门用于trend dependency约束的统计
    
    # 使用集合存储已违反的行索引，避免重复计数
    violation_rows_by_type = {}
    
    for violation in violations:
        v_type = violation['type']
        
        # 初始化集合
        if v_type not in violation_rows_by_type:
            violation_rows_by_type[v_type] = set()
        
        # 添加行索引到集合中
        if 'row_index' in violation:
            violation_rows_by_type[v_type].add(violation['row_index'])
        elif 'row_index_j' in violation:  # 特别处理OD约束
            violation_rows_by_type[v_type].add(violation['row_index_j'])
        
        # 对于列约束，额外统计每列的违反次数
        if v_type.startswith('column_'):
            column = violation.get('column', violation.get('column_index'))
            if column:
                if v_type not in column_violations_detail:
                    column_violations_detail[v_type] = {}
                if column not in column_violations_detail[v_type]:
                    column_violations_detail[v_type][column] = set()
                if 'row_index' in violation:
                    column_violations_detail[v_type][column].add(violation['row_index'])
        
        # 对于order dependency约束，统计涉及的变量对
        if v_type == "order_dependency":
            source_col = violation.get('source_column')
            target_col = violation.get('target_column')
            if source_col and target_col:
                od_pair = f"{source_col} → {target_col}"
                if v_type not in od_violations_detail:
                    od_violations_detail[v_type] = {}
                if od_pair not in od_violations_detail[v_type]:
                    od_violations_detail[v_type][od_pair] = set()
                # OD约束使用row_index_j作为违反行
                if 'row_index_j' in violation:
                    od_violations_detail[v_type][od_pair].add(violation['row_index_j'])
        
        # # 对于trend dependency约束，统计涉及的变量对
        # if v_type == "trend_dependency":
        #     source_col = violation.get('M')
        #     target_col = violation.get('N')
        #     if source_col and target_col:
        #         td_pair = f"{source_col} → {target_col}"
        #         if v_type not in td_violations_detail:
        #             td_violations_detail[v_type] = {}
        #         if td_pair not in td_violations_detail[v_type]:
        #             td_violations_detail[v_type][td_pair] = set()
        #         if 'row_index' in violation:
        #             td_violations_detail[v_type][td_pair].add(violation['row_index'])
    
    # 计算每种约束类型的违反行数
    for v_type, rows_set in violation_rows_by_type.items():
        violation_types[v_type] = len(rows_set)
    
    # 计算每种约束类型的违反率
    violation_rates = {}
    for v_type, count in violation_types.items():
        violation_rates[v_type] = {
            'count': count,
            'rate': count / total_rows
        }
    
    # 特别计算每列的违反率（针对列约束）
    for v_type, columns in column_violations_detail.items():
        violation_rates[v_type]['columns_detail'] = {}
        for column, rows_set in columns.items():
            count = len(rows_set)
            violation_rates[v_type]['columns_detail'][column] = {
                'count': count,
                'rate': count / total_rows
            }
    
    # 特别计算每个变量对的违反率（针对order dependency约束）
    for v_type, pairs in od_violations_detail.items():
        violation_rates[v_type]['pairs_detail'] = {}
        for pair, rows_set in pairs.items():
            count = len(rows_set)
            violation_rates[v_type]['pairs_detail'][pair] = {
                'count': count,
                'rate': count / total_rows
            }
    
    # # 特别计算每个变量对的违反率（针对trend dependency约束）
    # for v_type, pairs in td_violations_detail.items():
    #     violation_rates[v_type]['pairs_detail'] = {}
    #     for pair, rows_set in pairs.items():
    #         count = len(rows_set)
    #         violation_rates[v_type]['pairs_detail'][pair] = {
    #             'count': count,
    #             'rate': count / total_rows
    #         }
    
    # 计算总体违反率
    # 合并所有类型的违反行索引，避免重复计数
    all_violation_rows = set()
    for rows_set in violation_rows_by_type.values():
        all_violation_rows.update(rows_set)
    
    total_violations = len(all_violation_rows)
    violation_rates['total'] = {
        'count': total_violations,
        'rate': total_violations / total_rows
    }
    
    return violation_rates


def print_violation_rates(violation_rates):
    """
    打印违反率统计信息
    
    :param violation_rates: 违反率字典
    """
    if isinstance(violation_rates, str):
        print(violation_rates)
        return
    
    print("\n违反率统计:")
    print(f"  总体违反率: {violation_rates['total']['rate']:.4f} ({violation_rates['total']['count']} / {violation_rates['total'].get('count', 0) / violation_rates['total']['rate']:.0f})")
    
    for v_type, stats in violation_rates.items():
        if v_type == 'total':
            continue
        print(f"  {v_type} 违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
        
        # 如果有列的详细信息，也打印出来
        if 'columns_detail' in stats:
            print(f"    各列详细信息:")
            for column, col_stats in stats['columns_detail'].items():
                print(f"      {column}: {col_stats['rate']:.4f} ({col_stats['count']} 次违反)")
        
        # 如果有order dependency或trend dependency的详细信息，也打印出来
        if 'pairs_detail' in stats:
            print(f"    各变量对详细信息:")
            for pair, pair_stats in stats['pairs_detail'].items():
                print(f"      {pair}: {pair_stats['rate']:.4f} ({pair_stats['count']} 次违反)")


def constraint_violations_report(violations):
    # 输出结果
    print("约束违反检测结果:")
    print(f"总共发现 {len(violations)} 个违反")

    summary = get_violation_summary(violations)

    for i, violation in enumerate(violations[:]):
        print(f"  {i+1}. 类型: {violation['type']}")
        if "row_index" in violation:
            print(f"     行索引: {violation['row_index']}")
        if "columns" in violation:
            print(f"     列名: {violation['columns_involved']}")
        print(f"     违反程度: {violation.get('violation_degree', 'N/A')}")
        print()


# 使用示例
if __name__ == "__main__":

    # data = {
    #     "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    #     "C": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
    #     "D": [2, 8, 18, 32, 50, 72, 98, 128, 162, 200],
    # }
    # data = pd.DataFrame(data)

    np.random.seed(42)
    data = pd.DataFrame(
        {
            "timestamp": range(100),
            "temperature": np.random.normal(20, 5, 100),
            "humidity": np.random.uniform(30, 70, 100),
        }
    )

    # data = pd.read_csv("/home/yyy/TSC/TSClean/AutoClean/Datasets/IDF_Power/IDF_Power_Dirty.csv")
    # data = pd.read_csv("/home/yyy/TSC/TSClean/AutoClean/Datasets/IDF/idf.csv")

    attr_num = data.shape[1]
    constraints = mine_all_constraints(
        data,
        degree=3,
        attr_num=attr_num,
        window=20,
        strategy="auto",
        min_support=0.1,
        min_conf=0.9,
        confidence_threshold=0.7
    )

    constraint_report(constraints)

    print("开始检测...")

    # 创建违反检测器
    detector = ConstraintViolationDetector(constraints)

    # 检测违反
    violations = detector.detect_all_violations(data)
    print("violations:", violations)

    summary = get_violation_summary(violations)
    
    # 计算并打印违反率
    violation_rates = get_violation_rates(violations, data)
    print_violation_rates(violation_rates)

    # constraint_violations_report(violations)

    # print(detector.cv)
