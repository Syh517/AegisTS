import numpy as np
from sklearn.linear_model import LinearRegression

def imr_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    p: int = 3,
    delta: float = 1e-5,
    max_iter: int = 50
) -> np.ndarray:
    """
    优化后的 IMR 修复函数：
    1. 支持多特征：循环处理所有 n_features。
    2. 严格训练：确保 AR 模型的训练窗口内完全不含原始异常点（Clean Window）。
    3. 性能：使用 NumPy 向量化操作替代逐行 isfinite 检查。
    """
    data_repaired = data_abnormal.copy().astype(np.float64)
    n_samples, n_timesteps, n_features = data_abnormal.shape

    # 时间轴步长检查
    if n_timesteps <= p:
        return data_repaired

    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        if not np.any(is_abnormal):
            continue

        for c in range(n_features):
            y = data_repaired[s, :, c]
            
            # 初始填充：IMR 需要一个初始值，通常用线性插值
            mask_normal = ~is_abnormal
            if mask_normal.sum() < p + 1:
                continue
                
            # 只有异常点需要被不断迭代修复
            bad_idx = np.where(is_abnormal)[0]
            
            for it in range(max_iter):
                # 构建滑动窗口 X: (N-p, p), y_target: (N-p,)
                X_all = np.lib.stride_tricks.sliding_window_view(y, p)[:-1]
                y_target_all = y[p:]
                
                # 关键：训练集必须是“干净”的
                # 只有当目标点 y_t 和它之前的 p 个点全部不是原始异常时，才用于训练
                is_window_clean = ~np.lib.stride_tricks.sliding_window_view(is_abnormal, p)[:-1].any(axis=1)
                train_mask = is_window_clean & (~is_abnormal[p:])
                
                if train_mask.sum() < p + 1:
                    break

                X_train = X_all[train_mask]
                y_train = y_target_all[train_mask]
                
                # 过滤无效值 (NaN/Inf) - 向量化处理
                valid_rows = np.isfinite(y_train) & np.isfinite(X_train).all(axis=1)
                if not np.any(valid_rows):
                    break
                    
                # 训练 AR 模型
                model = LinearRegression()
                model.fit(X_train[valid_rows], y_train[valid_rows])
                
                # 预测所有点
                y_pred_full = np.zeros_like(y)
                y_pred_full[p:] = model.predict(X_all)
                
                # 更新并检查收敛
                y_old = y[is_abnormal].copy()
                y[is_abnormal] = y_pred_full[is_abnormal]
                
                # 计算最大变动量
                curr_diff = np.max(np.abs(y[is_abnormal] - y_old))
                if curr_diff < delta:
                    break

            data_repaired[s, :, c] = y

    return data_repaired