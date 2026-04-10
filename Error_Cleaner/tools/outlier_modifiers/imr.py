import numpy as np
from sklearn.linear_model import LinearRegression

def imr_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None,
    p: int = 3,
    delta: float = 1e-5,
    max_iter: int = 15
) -> np.ndarray:
    """
    自适应 IMR 修复函数（无 Label 版）：
    1. 自动定位：利用 AR 模型的预测残差自动识别时序上的突变噪声。
    2. 迭代收敛：通过多次拟合与预测，将噪声点逐步平滑至极大似然轨迹。
    3. 稳健拟合：在模型训练阶段自动过滤高残差样本点。
    """
    data_repaired = data_abnormal.copy().astype(np.float64)
    n_samples, n_timesteps, n_features = data_abnormal.shape

    if n_timesteps <= p:
        return data_repaired

    for s in range(n_samples):
        for c in range(n_features):
            y = data_repaired[s, :, c]
            
            # 为了自适应检测，我们需要维护一个“当前认定的异常掩码”
            # 初始时，可以基于简单的全局 3-sigma 识别
            mu_y, std_y = np.mean(y), np.std(y)
            current_is_bad = np.abs(y - mu_y) > 3 * std_y
            
            for it in range(max_iter):
                # 构造滑动窗口数据
                # X_all 形状: (n_timesteps - p, p)
                X_all = np.lib.stride_tricks.sliding_window_view(y, p)[:-1]
                y_target_all = y[p:]
                
                # --- 1. 稳健训练集选择 ---
                # 只选择那些当前不被认为是“坏”的点及其前驱窗口
                is_window_bad = np.lib.stride_tricks.sliding_window_view(current_is_bad, p)[:-1].any(axis=1)
                train_mask = (~is_window_bad) & (~current_is_bad[p:])
                
                # 如果正常点太少，适当放宽条件或终止迭代
                if train_mask.sum() < p + 2:
                    # 保底：若无法训练，则使用中值平滑辅助识别
                    break

                model = LinearRegression()
                model.fit(X_all[train_mask], y_target_all[train_mask])
                
                # --- 2. 全量预测与残差计算 ---
                y_pred_full = np.zeros_like(y)
                y_pred_full[p:] = model.predict(X_all)
                # 处理起始 p 个点（前向预测补齐）
                y_pred_full[:p] = y_pred_full[p:2*p][::-1] 
                
                residuals = np.abs(y - y_pred_full)
                # 计算动态阈值：残差的稳健标准差
                res_std = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
                
                # --- 3. 自动识别更新 ---
                # 只有残差显著大于噪声水平的点才被认为是异常
                new_is_bad = residuals > (3 * res_std + 1e-6)
                
                # --- 4. 迭代修复 ---
                y_old = y.copy()
                # 更新认定为坏的点：用预测值（似然值）替换原始观测值
                y[new_is_bad] = y_pred_full[new_is_bad]
                
                # 更新当前掩码供下一轮迭代使用
                current_is_bad = new_is_bad
                
                # 收敛检查
                max_change = np.max(np.abs(y - y_old))
                if max_change < delta:
                    break

            data_repaired[s, :, c] = y

    return data_repaired