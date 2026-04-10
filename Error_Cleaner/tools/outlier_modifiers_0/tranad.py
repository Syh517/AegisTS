import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings

def tranad_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window=10,
    epochs=15,
    batch_size=64,
    lr=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 1. 全局标准化 (嵌入函数内部)
    scaler = StandardScaler()
    normal_mask = (label == 0)
    if not np.any(normal_mask): return data_abnormal
    
    scaler.fit(data_abnormal[normal_mask])
    data_scaled = scaler.transform(data_abnormal.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # 2. 构建全局训练集：目标是预测当前点的值
    X_train_list, y_train_list = [], []
    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        for i in range(n_timestamps - window):
            # 确保窗口(前window个)和目标(当前点)全部为正常数据
            if not np.any(is_abnormal[i : i + window + 1]):
                X_train_list.append(data_scaled[s, i : i + window])
                y_train_list.append(data_scaled[s, i + window])

    if len(X_train_list) == 0:
        print("[TranAD] No clean windows found. Skip.")
        return data_abnormal

    X_train = torch.tensor(np.array(X_train_list))
    y_train = torch.tensor(np.array(y_train_list))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # 3. 定义模型 (MLP)
    class MLPReconstructor(nn.Module):
        def __init__(self, L, F):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(L * F, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, F)
            )
        def forward(self, x):
            return self.net(x.reshape(x.size(0), -1))

    model = MLPReconstructor(window, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. 全局训练一次
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 5. 递归修复 (自回归：使用已修复的点预测下一个点)
    data_repaired_scaled = data_scaled.copy()
    model.eval()
    with torch.no_grad():
        for s in range(n_samples):
            for t in range(n_timestamps):
                if label[s, t] == 1:
                    if t >= window:
                        # 核心逻辑：从 data_repaired_scaled 取上下文
                        ctx = data_repaired_scaled[s, t-window:t]
                        ctx_tensor = torch.tensor(ctx).unsqueeze(0).to(device)
                        data_repaired_scaled[s, t] = model(ctx_tensor).cpu().numpy()[0]
                    else:
                        # 开头部分异常，用 0 (即均值) 填充
                        data_repaired_scaled[s, t] = 0.0

    # 6. 逆变换还原
    repaired_final = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    return repaired_final