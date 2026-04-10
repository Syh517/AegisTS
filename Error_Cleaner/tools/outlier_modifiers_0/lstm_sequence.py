import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def lstm_sequence_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window=5,
    epochs=10,
    batch_size=64,
    hidden_dim=64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 1. 归一化处理
    # 注意：StandardScaler 需要 2D 输入 (N, n_features)
    scaler = StandardScaler()
    # 仅使用正常点来拟合 Scaler，防止异常点拉偏均值和方差
    normal_points_all = data_abnormal[label == 0]
    if len(normal_points_all) == 0:
        return data_abnormal # 无正常点无法修复
    
    scaler.fit(normal_points_all)
    
    # 将整个 3D 数组展平、转换、再还原回 3D
    data_reshaped = data_abnormal.reshape(-1, n_features)
    data_scaled = scaler.transform(data_reshaped).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # 2. 构造训练集 (使用归一化后的数据)
    X_train_list, y_train_list = [], []
    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        for i in range(n_timestamps - window):
            # 只有全正常的窗口才入库
            if not np.any(is_abnormal[i : i + window + 1]):
                X_train_list.append(data_scaled[s, i : i + window])
                y_train_list.append(data_scaled[s, i + window])

    if len(X_train_list) == 0:
        return data_abnormal

    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)

    # 3. 模型定义
    class SequenceLSTM(nn.Module):
        def __init__(self, input_dim, hidden):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2)
            self.fc = nn.Linear(hidden, input_dim)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    model = SequenceLSTM(n_features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 4. 训练
    model.train()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 5. 修复逻辑 (在 Scaled 空间进行)
    data_repaired_scaled = data_scaled.copy()
    model.eval()
    with torch.no_grad():
        for s in range(n_samples):
            for t in range(n_timestamps):
                if label[s, t] == 1:
                    if t >= window:
                        input_seq = data_repaired_scaled[s, t-window:t]
                        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
                        pred = model(input_tensor).cpu().numpy()[0]
                        data_repaired_scaled[s, t] = pred
                    else:
                        # 序列头部的异常点，用 0 填充（标准分布的均值是 0）
                        data_repaired_scaled[s, t] = 0.0

    # 6. 逆变换：还原量纲
    repaired_reshaped = data_repaired_scaled.reshape(-1, n_features)
    final_data = scaler.inverse_transform(repaired_reshaped).reshape(n_samples, n_timestamps, n_features)

    return final_data