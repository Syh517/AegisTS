import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

def bayesian_model_repair(
    data_abnormal: np.ndarray, label: np.ndarray
) -> np.ndarray:
    """
    优化后的贝叶斯模型修复函数：
    1. 性能优化：通过预检查减少不必要的模型拟合。
    2. 数值稳定性：引入时间轴标准化，防止大时间戳导致的权重失衡。
    3. 异常处理：增加了针对正常点不足情况的自动降级机制。
    """
    data_repaired = data_abnormal.copy()
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 预处理时间轴特征
    x = np.arange(n_timestamps).reshape(-1, 1)
    # 标准化时间戳：将 (0, 1, 2...) 变为均值为0方差为1的分布，有助于贝叶斯推断
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    
    # 提前初始化模型对象（复用对象，只改变训练数据，减少开销）
    model = BayesianRidge()

    for i in range(n_samples):
        is_abnormal = (label[i] == 1)
        if not np.any(is_abnormal):
            continue
            
        for j in range(n_features):
            y = data_abnormal[i, :, j]
            mask_normal = ~is_abnormal
            
            # 基础校验：贝叶斯岭回归至少需要 2 个点才能确定直线
            n_normal = np.sum(mask_normal)
            if n_normal < 2:
                # 极端情况回退：使用均值填充
                avg = np.nanmean(y) if n_normal > 0 else 0.0
                data_repaired[i, is_abnormal, j] = avg
                continue

            try:
                # 拟合正常点
                model.fit(x_scaled[mask_normal], y[mask_normal])
                
                # 仅预测异常点所在的位置，显著减少推理开销
                # 形状：(n_abnormal_points, 1)
                x_pred = x_scaled[is_abnormal]
                y_pred = model.predict(x_pred)
                
                data_repaired[i, is_abnormal, j] = y_pred
                
            except Exception:
                # 拟合失败回退：简单的线性插值
                data_repaired[i, is_abnormal, j] = np.interp(
                    x[is_abnormal].ravel(), x[mask_normal].ravel(), y[mask_normal]
                )
                
    return data_repaired