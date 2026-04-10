import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def svr_repair(data_abnormal: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    优化后的 SVR 修复函数：
    1. 引入 StandardScaler：解决 SVR 对量程敏感的问题，大幅提升拟合准确度。
    2. 减少冗余计算：预检查异常点，避免对无异常的特征进行耗时的 SVR 训练。
    3. 优化超参数：增加并行执行建议（若样本量极大）。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape

    # 时间索引归一化（有助于 SVR 稳定收敛）
    x = np.arange(n_timestamps).reshape(-1, 1)
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)

    for sample in range(n_samples):
        # 预检查：如果该样本完全没有异常，直接跳过
        sample_label = label[sample]
        if not np.any(sample_label == 1):
            continue

        for col in range(n_features):
            y = data_abnormal[sample, :, col]
            mask_normal = (sample_label == 0)
            mask_abnormal = (sample_label == 1)

            # 只有当该特征存在异常点时才进行 SVR
            if not np.any(mask_abnormal):
                continue

            # 基础校验：正常点太少则使用线性插值
            if np.sum(mask_normal) < 5:  # SVR 建议至少 5 个点以捕捉 RBF 形状
                s_pd = pd.Series(y)
                s_pd[mask_abnormal] = np.nan
                data_repaired[sample, mask_abnormal, col] = (
                    s_pd.interpolate(method='linear')
                    .fillna(method="bfill")
                    .fillna(method="ffill")
                ).values[mask_abnormal]
                continue

            try:
                # 核心改进：对 Y 进行标准化，防止 C 和 epsilon 在不同量程下失效
                y_normal = y[mask_normal].reshape(-1, 1)
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_normal).ravel()

                # 配置 SVR：
                # cache_size=1000 增加内存利用减少磁盘IO
                # C 和 epsilon 需要根据数据噪声微调
                svr = SVR(kernel="rbf", C=10.0, epsilon=0.05, cache_size=1000)
                svr.fit(x_scaled[mask_normal], y_train_scaled)
                
                # 仅预测异常位置，减少推理时间
                y_pred_scaled = svr.predict(x_scaled[mask_abnormal])
                
                # 反标准化回原始量程
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                data_repaired[sample, mask_abnormal, col] = y_pred

            except Exception:
                # 拟合失败的回退机制
                y_interp = np.interp(x[mask_abnormal].ravel(), x[mask_normal].ravel(), y[mask_normal])
                data_repaired[sample, mask_abnormal, col] = y_interp

    return data_repaired