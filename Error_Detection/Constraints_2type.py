import numpy as np
import pandas as pd
import re
from typing import Union, List, Dict, Any

from Error_Detection.miner_2type_row_liner_col import mine_all_constraints_per_sample, constraint_report
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
        :param constraints: 从mine_all_constraints函数得到的所有约束，或者从mine_all_constraints_per_sample得到的每个样本的约束
        """
        self.constraints = constraints
        self.violations = []
        self.time_col = "timestamp"  # timestamp列保护
        self.is_3d = False
        self.n_samples = 1
        self.cv_matrices = []
        # Check if constraints are per-sample constraints
        self.is_per_sample = isinstance(constraints, list) and len(constraints) > 0 and "sample_index" in constraints[0]

    def _prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        准备数据，支持2D和3D数据格式
        :param data: 输入数据，可以是DataFrame或numpy数组
        :return: 处理后的DataFrame或DataFrame列表
        """
        n_features = data.shape[-1]
        if isinstance(data, np.ndarray):
            # 处理3D数据 (n_samples, n_timestamps, n_features)
            if data.ndim == 3:
                self.is_3d = True
                self.n_samples = data.shape[0]
                # 为每个样本创建DataFrame
                dataframes = []
                for i in range(self.n_samples):
                    # 生成列名
                    columns = ["timestamp"] + [f"col_{j}" for j in range(1, n_features)]
                    df = pd.DataFrame(data[i], columns=columns)
                    dataframes.append(df)
                return dataframes
            # 处理2D数组 (n_timestamps, n_features)
            elif data.ndim == 2:
                columns = [f"col_{i}" for i in range(data.shape[1])]
                return pd.DataFrame(data, columns=columns)
            else:
                raise ValueError("数据必须是2D或3D数组")
        elif isinstance(data, pd.DataFrame):
            # 已经是DataFrame格式
            return data.copy()
        else:
            raise TypeError("数据必须是numpy数组或pandas DataFrame")

    def _prepare_cv_matrix(self, data: pd.DataFrame):
        """
        为当前数据准备CV矩阵
        :param data: 输入的DataFrame数据
        """
        self.cv = pd.DataFrame(0.0, index=data.index, columns=data.columns, dtype=float)

    def detect_column_violations(self, data: pd.DataFrame, sample_index: int = 0) -> List[Dict]:
        """
        检测列约束违反（速度、加速度、方差、振幅）
        :param data: 待检测的数据
        :param sample_index: 样本索引（用于3D数据）
        :return: 列约束违反列表
        """
        column_violations = []
        
        # Get constraints for this sample if using per-sample constraints
        if self.is_per_sample:
            sample_constraints = self.constraints[sample_index]
            speed_constraints = sample_constraints.get("speed_constraints", {})
            accel_constraints = sample_constraints.get("accel_constraints", {})
            variance_constraints = sample_constraints.get("variance_constraints", {})
            amplitude_constraints = sample_constraints.get("amplitude_constraints", {})
        else:
            speed_constraints = self.constraints.get("speed_constraints", {})
            accel_constraints = self.constraints.get("accel_constraints", {})
            variance_constraints = self.constraints.get("variance_constraints", {})
            amplitude_constraints = self.constraints.get("amplitude_constraints", {})

        for col in data.columns:
            # 跳过timestamp列的约束检测
            if col == self.time_col:
                continue
                
            series = data[col]

            # 速度约束检测
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
                        violation_record = {
                            "type": "column_speed",
                            "column": col,
                            "column_index": col,
                            "row_index": idx,
                            "value": value,
                            "lower_bound": speed_range[0],
                            "upper_bound": speed_range[1],
                            "violation_degree": violation_degree,
                            "columns_involved": [col],
                            "sample_index": sample_index
                        }
                        column_violations.append(violation_record)
                        self.cv.loc[idx, col] += violation_degree

            # 加速度约束检测
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
                        violation_record = {
                            "type": "column_acceleration",
                            "column": col,
                            "column_index": col,
                            "row_index": idx,
                            "value": value,
                            "lower_bound": accel_range[0],
                            "upper_bound": accel_range[1],
                            "violation_degree": violation_degree,
                            "columns_involved": [col],
                            "sample_index": sample_index
                        }
                        column_violations.append(violation_record)
                        self.cv.loc[idx, col] += violation_degree

            # 方差约束检测
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
                        violation_record = {
                            "type": "column_variance",
                            "column": col,
                            "column_index": col,
                            "row_index": i + window_size // 2,
                            "value": local_variance,
                            "lower_bound": variance_range[0],
                            "upper_bound": variance_range[1],
                            "violation_degree": violation_degree,
                            "columns_involved": [col],
                            "sample_index": sample_index
                        }
                        column_violations.append(violation_record)
                        for window_idx in range(i, i + window_size):
                            self.cv.loc[window_idx, col] += violation_degree

            # 振幅约束检测
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
                    violation_record = {
                        "type": "column_amplitude",
                        "column": col,
                        "column_index": col,
                        "row_index": idx,
                        "value": value,
                        "lower_bound": amplitude_range[0],
                        "upper_bound": amplitude_range[1],
                        "violation_degree": violation_degree,
                        "columns_involved": [col],
                        "sample_index": sample_index
                    }
                    column_violations.append(violation_record)
                    self.cv.loc[idx, col] += violation_degree

        return column_violations

    def detect_row_violations(self, data: pd.DataFrame, sample_index: int = 0) -> List[Dict]:
        """
        检测数据行是否违反"多变量关系型"行约束
        约束形式：target ≈ f(features)，残差 ∈ [lower, upper]
        :param data: 待检测的数据
        :param sample_index: 样本索引（用于3D数据）
        :return: 行约束违反列表
        """
        row_violations = []
        
        # Get constraints for this sample if using per-sample constraints
        if self.is_per_sample:
            sample_constraints = self.constraints[sample_index]
            row_constraints = sample_constraints.get("row_constraints", [])
        else:
            row_constraints = self.constraints.get("row_constraints", [])
            
        if not row_constraints:
            return row_violations

        for i, constraint in enumerate(row_constraints):
            formula = constraint["formula"]
            target = constraint["target"]
            features = constraint["features"]
            coef = constraint.get("coef", {})  # 兼容旧格式
            model_terms = constraint.get("model_terms", {})  # 新格式
            # 优先使用model_terms，如果没有则使用coef
            if not coef and model_terms:
                coef = model_terms
            intercept = constraint.get("intercept", 0)
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
                violation_record = {
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
                    "sample_index": sample_index
                }
                row_violations.append(violation_record)

                # 累加违反度到 cv（可选：加权或平方）
                for col in [target] + features:
                    if col != self.time_col:
                        self.cv.loc[idx, col] += violation_degree

        return row_violations

    def detect_od_violations(self, data: pd.DataFrame, sample_index: int = 0) -> List[Dict]:
        """
        检测 Order Dependency 约束违反（归并排序 O(n log n)）
        返回违反的行对信息
        :param data: 待检测的数据
        :param sample_index: 样本索引（用于3D数据）
        :return: OD约束违反列表
        """
        od_violations = []
        
        # Get constraints for this sample if using per-sample constraints
        if self.is_per_sample:
            sample_constraints = self.constraints[sample_index]
            od_constraints = sample_constraints.get("od_constraints", [])
        else:
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
                    violation_record = {
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
                        "sample_index": sample_index
                    }
                    od_violations.append(violation_record)
                    if target_col != self.time_col:
                        self.cv.loc[row_j, target_col] += violation_degree
                    recorded_rows.add(row_j)
        return od_violations

    def detect_dd_violations(self, data: pd.DataFrame, sample_index: int = 0) -> List[Dict]:
        """
        检测 Denial Dependency 约束违反
        :param data: 待检测的数据
        :param sample_index: 样本索引（用于3D数据）
        :return: DD约束违反列表
        """
        dd_violations = []
        
        # Get constraints for this sample if using per-sample constraints
        if self.is_per_sample:
            sample_constraints = self.constraints[sample_index]
            dd_constraints = sample_constraints.get("dd_constraints", [])
        else:
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
                    violation_record = {
                        "type": "denial_dependency",
                        "constraint_index": i,
                        "row_index": idx,
                        "formula": dd_formula,
                        "support": support,
                        "confidence": confidence,
                        "violation_degree": violation_degree,
                        "columns_involved": list(columns_involved),
                        "sample_index": sample_index
                    }
                    dd_violations.append(violation_record)
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
                    violation_record = {
                        "type": "denial_dependency",
                        "constraint_index": i,
                        "row_index": idx,
                        "formula": dd_formula,
                        "support": support,
                        "confidence": confidence,
                        "violation_degree": violation_degree,
                        "columns_involved": list(columns_involved),
                        "sample_index": sample_index
                    }
                    dd_violations.append(violation_record)
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
                violation_record = {
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
                    "sample_index": sample_index
                }
                dd_violations.append(violation_record)
        return dd_violations

    def detect_all_violations(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        检测所有类型的约束违反
        
        :param data: 待检测的数据，可以是DataFrame或numpy数组（2D或3D）
        :return: 所有违反的列表
        """
        # 预处理数据
        processed_data = self._prepare_data(data)
        
        all_violations = []
        
        if self.is_3d:
            # 处理3D数据，对每个样本分别检测
            self.cv_matrices = []
            for i, df in enumerate(processed_data):
                # 为当前样本初始化cv矩阵
                self._prepare_cv_matrix(df)
                
                # print(f"正在处理样本 {i+1}/{self.n_samples}")
                
                # 对每个样本分别检测各类约束
                sample_violations = []
                sample_violations.extend(self.detect_column_violations(df, i))
                sample_violations.extend(self.detect_row_violations(df, i))
                sample_violations.extend(self.detect_od_violations(df, i))
                sample_violations.extend(self.detect_dd_violations(df, i))
                
                all_violations.extend(sample_violations)
                # 保存当前样本的cv矩阵
                self.cv_matrices.append(self.cv.copy())
        else:
            # 处理2D数据
            self._prepare_cv_matrix(processed_data)
            all_violations.extend(self.detect_column_violations(processed_data))
            all_violations.extend(self.detect_row_violations(processed_data))
            all_violations.extend(self.detect_od_violations(processed_data))
            all_violations.extend(self.detect_dd_violations(processed_data))
            self.cv_matrices = [self.cv]  # 即使是2D数据也保持一致性
        
        self.violations = all_violations
        return all_violations

    def calculate_violation_rates(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        计算各种约束类型的违反率
        
        :param data: 原始数据DataFrame或numpy数组
        :return: 包含各种约束违反率的字典
        """
        if not hasattr(self, 'violations') or not self.violations:
            raise ValueError("请先运行detect_all_violations方法检测违反情况")
        
        # 预处理数据以获取正确的行数
        processed_data = self._prepare_data(data)
        
        if self.is_3d:
            # 对于3D数据，计算所有样本的总行数
            total_rows = sum(len(df) for df in processed_data)
        else:
            # 对于2D数据
            total_rows = len(processed_data)
        
        # 初始化违反率统计
        violation_rates = {}
        
        # 按类型分组统计违反次数，使用集合去重以避免重复计算同一行
        violation_types = {}
        column_violations = {}  # 专门用于列约束的统计
        od_violations_detail = {}  # 专门用于order dependency约束的统计
        row_violations_detail = {}  # 专门用于行约束的统计
        dd_violations_detail = {}  # 专门用于denial dependency约束的统计
        
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
            
            # 对于行约束，统计涉及的公式
            if v_type == "row_constraint":
                formula = violation.get('formula')
                if formula:
                    if v_type not in row_violations_detail:
                        row_violations_detail[v_type] = {}
                    if formula not in row_violations_detail[v_type]:
                        row_violations_detail[v_type][formula] = set()
                    if 'row_index' in violation:
                        row_violations_detail[v_type][formula].add(violation['row_index'])
            
            # 对于denial dependency约束，统计涉及的公式
            if v_type == "denial_dependency":
                formula = violation.get('formula')
                if formula:
                    if v_type not in dd_violations_detail:
                        dd_violations_detail[v_type] = {}
                    if formula not in dd_violations_detail[v_type]:
                        dd_violations_detail[v_type][formula] = set()
                    if 'row_index' in violation:
                        dd_violations_detail[v_type][formula].add(violation['row_index'])
            
        # 计算每种约束类型的违反行数
        for v_type, rows_set in violation_rows_by_type.items():
            violation_types[v_type] = len(rows_set)
        
        # 计算每种约束类型的违反率
        for v_type, count in violation_types.items():
            violation_rates[v_type] = {
                'count': count,
                'rate': count / total_rows
            }
            
            # 为列约束类型添加详细的列级统计
            if v_type.startswith('column_'):
                violation_rates[v_type]['columns'] = {}
                if v_type in column_violations:
                    for column, rows_set in column_violations[v_type].items():
                        col_count = len(rows_set)
                        violation_rates[v_type]['columns'][column] = {
                            'count': col_count,
                            'rate': col_count / total_rows
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
        
        # 特别计算每个公式的违反率（针对行约束）
        for v_type, formulas in row_violations_detail.items():
            violation_rates[v_type]['formulas_detail'] = {}
            for formula, rows_set in formulas.items():
                count = len(rows_set)
                violation_rates[v_type]['formulas_detail'][formula] = {
                    'count': count,
                    'rate': count / total_rows
                }
        
        # 特别计算每个公式的违反率（针对denial dependency约束）
        for v_type, formulas in dd_violations_detail.items():
            violation_rates[v_type]['formulas_detail'] = {}
            for formula, rows_set in formulas.items():
                count = len(rows_set)
                violation_rates[v_type]['formulas_detail'][formula] = {
                    'count': count,
                    'rate': count / total_rows
                }
        
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

    def calculate_sample_violation_rates(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[int, Dict]:
        """
        计算每个样本的违反率
        
        :param data: 原始数据DataFrame或numpy数组
        :return: 每个样本的违反率字典，键为样本索引，值为该样本的违反率统计
        """
        if not hasattr(self, 'violations') or not self.violations:
            raise ValueError("请先运行detect_all_violations方法检测违反情况")
        
        # 预处理数据以获取正确的行数
        processed_data = self._prepare_data(data)
        
        if not self.is_3d:
            # 对于2D数据，直接返回整体违反率
            return {0: self.calculate_violation_rates(data)}
        
        # 对于3D数据，分别计算每个样本的违反率
        sample_violation_rates = {}
        
        # 按样本分组违反记录
        violations_by_sample = {}
        for violation in self.violations:
            sample_idx = violation.get('sample_index', 0)
            if sample_idx not in violations_by_sample:
                violations_by_sample[sample_idx] = []
            violations_by_sample[sample_idx].append(violation)
        
        # 分别计算每个样本的违反率
        for sample_idx, sample_violations in violations_by_sample.items():
            # 获取样本数据
            sample_data = processed_data[sample_idx]
            total_rows = len(sample_data)
            
            # 初始化违反率统计
            violation_rates = {}
            
            # 按类型分组统计违反次数
            violation_types = {}
            speed_violations = {}      # 专门用于速度约束的统计
            accel_violations = {}      # 专门用于加速度约束的统计
            variance_violations = {}   # 专门用于方差约束的统计
            amplitude_violations = {}  # 专门用于振幅约束的统计
            od_violations_detail = {}  # 专门用于order dependency约束的统计
            row_violations_detail = {} # 专门用于行约束的统计
            dd_violations_detail = {}  # 专门用于denial dependency约束的统计
            
            # 使用集合存储已违反的行索引，避免重复计数
            violation_rows_by_type = {}
            
            for violation in sample_violations:
                v_type = violation['type']
                
                # 初始化集合
                if v_type not in violation_rows_by_type:
                    violation_rows_by_type[v_type] = set()
                
                # 添加行索引到集合中
                if 'row_index' in violation:
                    violation_rows_by_type[v_type].add(violation['row_index'])
                elif 'row_index_j' in violation:  # 特别处理OD约束
                    violation_rows_by_type[v_type].add(violation['row_index_j'])
                
                # 对于不同类型的列约束，分别统计
                column = violation.get('column', violation.get('column_index'))
                if column:
                    if v_type == "column_speed":
                        if v_type not in speed_violations:
                            speed_violations[v_type] = {}
                        if column not in speed_violations[v_type]:
                            speed_violations[v_type][column] = set()
                        if 'row_index' in violation:
                            speed_violations[v_type][column].add(violation['row_index'])
                            
                    elif v_type == "column_acceleration":
                        if v_type not in accel_violations:
                            accel_violations[v_type] = {}
                        if column not in accel_violations[v_type]:
                            accel_violations[v_type][column] = set()
                        if 'row_index' in violation:
                            accel_violations[v_type][column].add(violation['row_index'])
                            
                    elif v_type == "column_variance":
                        if v_type not in variance_violations:
                            variance_violations[v_type] = {}
                        if column not in variance_violations[v_type]:
                            variance_violations[v_type][column] = set()
                        if 'row_index' in violation:
                            variance_violations[v_type][column].add(violation['row_index'])
                            
                    elif v_type == "column_amplitude":
                        if v_type not in amplitude_violations:
                            amplitude_violations[v_type] = {}
                        if column not in amplitude_violations[v_type]:
                            amplitude_violations[v_type][column] = set()
                        if 'row_index' in violation:
                            amplitude_violations[v_type][column].add(violation['row_index'])
                
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
                
                # 对于行约束，统计涉及的公式
                if v_type == "row_constraint":
                    formula = violation.get('formula')
                    if formula:
                        if v_type not in row_violations_detail:
                            row_violations_detail[v_type] = {}
                        if formula not in row_violations_detail[v_type]:
                            row_violations_detail[v_type][formula] = set()
                        if 'row_index' in violation:
                            row_violations_detail[v_type][formula].add(violation['row_index'])
                
                # 对于denial dependency约束，统计涉及的公式
                if v_type == "denial_dependency":
                    formula = violation.get('formula')
                    if formula:
                        if v_type not in dd_violations_detail:
                            dd_violations_detail[v_type] = {}
                        if formula not in dd_violations_detail[v_type]:
                            dd_violations_detail[v_type][formula] = set()
                        if 'row_index' in violation:
                            dd_violations_detail[v_type][formula].add(violation['row_index'])
            
            # 计算每种约束类型的违反行数
            for v_type, rows_set in violation_rows_by_type.items():
                violation_types[v_type] = len(rows_set)
            
            # 计算每种约束类型的违反率
            for v_type, count in violation_types.items():
                violation_rates[v_type] = {
                    'count': count,
                    'rate': count / total_rows
                }
            
            # 特别计算每列的违反率（针对速度约束）
            if "column_speed" in violation_types and "column_speed" in speed_violations:
                violation_rates["column_speed"]['columns'] = {}
                for column, rows_set in speed_violations["column_speed"].items():
                    count = len(rows_set)
                    violation_rates["column_speed"]['columns'][column] = {
                        'count': count,
                        'rate': count / total_rows
                    }
                    
            # 特别计算每列的违反率（针对加速度约束）
            if "column_acceleration" in violation_types and "column_acceleration" in accel_violations:
                violation_rates["column_acceleration"]['columns'] = {}
                for column, rows_set in accel_violations["column_acceleration"].items():
                    count = len(rows_set)
                    violation_rates["column_acceleration"]['columns'][column] = {
                        'count': count,
                        'rate': count / total_rows
                    }
                    
            # 特别计算每列的违反率（针对方差约束）
            if "column_variance" in violation_types and "column_variance" in variance_violations:
                violation_rates["column_variance"]['columns'] = {}
                for column, rows_set in variance_violations["column_variance"].items():
                    count = len(rows_set)
                    violation_rates["column_variance"]['columns'][column] = {
                        'count': count,
                        'rate': count / total_rows
                    }
                    
            # 特别计算每列的违反率（针对振幅约束）
            if "column_amplitude" in violation_types and "column_amplitude" in amplitude_violations:
                violation_rates["column_amplitude"]['columns'] = {}
                for column, rows_set in amplitude_violations["column_amplitude"].items():
                    count = len(rows_set)
                    violation_rates["column_amplitude"]['columns'][column] = {
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
            
            # 特别计算每个公式的违反率（针对行约束）
            for v_type, formulas in row_violations_detail.items():
                violation_rates[v_type]['formulas_detail'] = {}
                for formula, rows_set in formulas.items():
                    count = len(rows_set)
                    violation_rates[v_type]['formulas_detail'][formula] = {
                        'count': count,
                        'rate': count / total_rows
                    }
            
            # 特别计算每个公式的违反率（针对denial dependency约束）
            for v_type, formulas in dd_violations_detail.items():
                violation_rates[v_type]['formulas_detail'] = {}
                for formula, rows_set in formulas.items():
                    count = len(rows_set)
                    violation_rates[v_type]['formulas_detail'][formula] = {
                        'count': count,
                        'rate': count / total_rows
                    }
            
            # 计算总体违反率
            all_violation_rows = set()
            for rows_set in violation_rows_by_type.values():
                all_violation_rows.update(rows_set)
            
            total_violations = len(all_violation_rows)
            violation_rates['total'] = {
                'count': total_violations,
                'rate': total_violations / total_rows
            }
            
            sample_violation_rates[sample_idx] = violation_rates
            
        return sample_violation_rates

    def print_sample_violation_rates(self, sample_violation_rates: Dict[int, Dict]):
        """
        按照指定格式打印每个样本的违反率并返回每个样本每种约束的违反率
        
        :param sample_violation_rates: 每个样本的违反率字典
        :return: 每个样本每种约束的违反率字典，约束类型直接对应违反率值
        """
        print("\n=== 各样本约束违反率 ===")
        # 用于存储每个样本每种约束的违反率
        sample_constraint_violation_rates = {}
        
        for sample_idx, rates in sample_violation_rates.items():
            # print(f"\n样本 {sample_idx}:")
            sample_constraint_violation_rates[sample_idx] = {}
            for v_type, stats in rates.items():
                # if v_type == 'total':
                #     print(f"  总体违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                # elif v_type == 'column_speed' and 'columns' in stats:
                #     print(f"  速度约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for column, col_stats in stats['columns'].items():
                #     #     print(f"    {column}: {col_stats['rate']:.4f} ({col_stats['count']} 次违反)")
                # elif v_type == 'column_acceleration' and 'columns' in stats:
                #     print(f"  加速度约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for column, col_stats in stats['columns'].items():
                #     #     print(f"    {column}: {col_stats['rate']:.4f} ({col_stats['count']} 次违反)")
                # elif v_type == 'column_variance' and 'columns' in stats:
                #     print(f"  方差约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for column, col_stats in stats['columns'].items():
                #     #     print(f"    {column}: {col_stats['rate']:.4f} ({col_stats['count']} 次违反)")
                # # elif v_type == 'column_amplitude' and 'columns' in stats:
                # #     print(f"  振幅约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                # #     # for column, col_stats in stats['columns'].items():
                # #     #     print(f"    {column}: {col_stats['rate']:.4f} ({col_stats['count']} 次违反)")
                        
                # elif v_type == 'order_dependency' and 'pairs_detail' in stats:
                #     print(f"  顺序依赖约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for pair, pair_stats in stats['pairs_detail'].items():
                #     #     print(f"    {pair}: {pair_stats['rate']:.4f} ({pair_stats['count']} 次违反)")

                # elif v_type == 'row_constraint' and 'formulas_detail' in stats:
                #     print(f"  行约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for formula, formula_stats in stats['formulas_detail'].items():
                #     #     print(f"    {formula}: {formula_stats['rate']:.4f} ({formula_stats['count']} 次违反)")
                # elif v_type == 'denial_dependency' and 'formulas_detail' in stats:
                #     print(f"  否认依赖约束违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                #     # for formula, formula_stats in stats['formulas_detail'].items():
                #     #     print(f"    {formula}: {formula_stats['rate']:.4f} ({formula_stats['count']} 次违反)")
                # else:
                #     print(f"  {v_type} 违反率: {stats['rate']:.4f} ({stats['count']} 次违反)")
                
                # 存储每个样本每种约束的违反率，直接存储违反率值
                # 将v_type名称映射为与constraints中对应的键名
                if v_type == 'column_speed':
                    mapped_type = 'speed_constraints'
                elif v_type == 'column_acceleration':
                    mapped_type = 'accel_constraints'
                elif v_type == 'column_variance':
                    mapped_type = 'variance_constraints'
                elif v_type == 'order_dependency':
                    mapped_type = 'od_constraints'
                elif v_type == 'row_constraint':
                    mapped_type = 'row_constraints'
                elif v_type == 'denial_dependency':
                    mapped_type = 'dd_constraints'
                else:
                    mapped_type = v_type
                    
                sample_constraint_violation_rates[sample_idx][mapped_type] = stats['rate']
        
        return sample_constraint_violation_rates

    def get_cv_matrices(self) -> List[pd.DataFrame]:
        """
        获取所有样本的违反程度矩阵
        
        :return: 违反程度矩阵列表，每个元素对应一个样本
        """
        if not hasattr(self, 'cv_matrices') or not self.cv_matrices:
            raise ValueError("请先运行detect_all_violations方法")
        return self.cv_matrices

# 使用示例
if __name__ == "__main__":
    # 3D数据处理示例
    print("\n=== 3D数据处理示例 ===")
    
    # 创建3D数据 (3个样本，每个样本有50个时间步，2个特征)
    # 模拟3个不同设备的温度和湿度数据
    np.random.seed(123)
    n_samples, n_timestamps, n_features = 3, 50, 2
    data_3d = np.zeros((n_samples, n_timestamps, n_features))
    
    for i in range(n_samples):
        # 每个样本有略微不同的统计特性
        data_3d[i, :, 0] = np.random.normal(20 + i, 2 + i/2, n_timestamps)  # 温度特征
        data_3d[i, :, 1] = np.random.uniform(30 + i, 70 + i, n_timestamps)  # 湿度特征
    
    # 为3D数据创建约束条件
    # 定义简单的约束规则
    constraints_3d = {
        "speed_constraints": {
            "col_0": (-5, 5),    # 温度变化速度约束
            "col_1": (-10, 10)   # 湿度变化速度约束
        },
    }
    
    print(f"3D数据形状: {data_3d.shape}")
    print(f"样本数量: {n_samples}")
    print(f"时间步数: {n_timestamps}")
    print(f"特征数: {n_features}")






    
    # 创建检测器并检测违反
    detector_3d = ConstraintViolationDetector(constraints_3d)
    violations_3d = detector_3d.detect_all_violations(data_3d)
    
    print(f"\n总共发现 {len(violations_3d)} 个违反")
    
    # # 按样本统计违反情况
    # violations_by_sample = {}
    # for violation in violations_3d:
    #     sample_idx = violation.get('sample_index', 0)
    #     if sample_idx not in violations_by_sample:
    #         violations_by_sample[sample_idx] = []
    #     violations_by_sample[sample_idx].append(violation)
    
    # for sample_idx, sample_violations in violations_by_sample.items():
    #     print(f"样本 {sample_idx} 中发现 {len(sample_violations)} 个违反")
    
    # # 获取违反程度矩阵
    # cv_matrices = detector_3d.get_cv_matrices()
    # print(f"\n获取到 {len(cv_matrices)} 个CV矩阵")
    
    # for i, cv_matrix in enumerate(cv_matrices):
    #     print(f"样本 {i} 的CV矩阵形状: {cv_matrix.shape}")
    #     # 显示部分CV矩阵数据
    #     print(f"样本 {i} 的CV矩阵前5行:")
    #     print(cv_matrix.head())
    
    # 计算违反率
    try:
        # violation_rates_3d = detector_3d.calculate_violation_rates(data_3d)
        # print("\n3D数据违反率统计:")
        # print(violation_rates_3d)
        
        # 计算每个样本的违反率
        sample_violation_rates = detector_3d.calculate_sample_violation_rates(data_3d)
        # 按指定格式输出
        detector_3d.print_sample_violation_rates(sample_violation_rates)
    except Exception as e:
        print(f"计算违反率时出错: {e}")