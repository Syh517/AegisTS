import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings

def cae_m_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    window: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    latent_dim: int = 8,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 1. 全局标准化：Autoencoder 必须进行标准化以保证 MSELoss 权重平衡
    scaler = StandardScaler()
    normal_mask_all = (label == 0)
    if not np.any(normal_mask_all): return data_abnormal
    
    scaler.fit(data_abnormal[normal_mask_all])
    data_scaled = scaler.transform(data_abnormal.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # 2. 构建全局重构训练集 (仅使用全正常窗口)
    X_train_list = []
    for s in range(n_samples):
        is_abnormal = (label[s] == 1)
        for i in range(n_timestamps - window + 1):
            if not np.any(is_abnormal[i : i + window]):
                X_train_list.append(data_scaled[s, i : i + window])

    if len(X_train_list) == 0:
        return data_abnormal

    X_train_tensor = torch.tensor(np.array(X_train_list))
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # 3. 定义模型 (移除 Sigmoid，增加层规范化)
    class MLPAE(nn.Module):
        def __init__(self, seq_len, n_features, latent_dim):
            super().__init__()
            input_dim = seq_len * n_features
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, input_dim) # 移除 Sigmoid，直接输出线性值
            )

        def forward(self, x):
            B, L, F = x.shape
            encoded = self.encoder(x.view(B, -1))
            decoded = self.decoder(encoded).view(B, L, F)
            return decoded

    model = MLPAE(window, n_features, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. 全局训练
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

    # 5. 修复逻辑：利用“重构”属性
    # 我们将异常点所在的窗口放入 AE，AE 会将其重构为最接近正常模式的序列
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            # 使用滑动窗口修复：如果窗口中心或结尾是异常，用重构值替换
            for t in range(n_timestamps - window + 1):
                win_slice = data_repaired_scaled[s, t : t + window]
                win_label = label[s, t : t + window]
                
                if np.any(win_label == 1):
                    win_tensor = torch.tensor(win_slice).unsqueeze(0).to(device)
                    reconstructed = model(win_tensor).cpu().numpy()[0]
                    
                    # 仅覆盖窗口中的异常点
                    for i in range(window):
                        if win_label[i] == 1:
                            data_repaired_scaled[s, t + i] = reconstructed[i]

    # 6. 逆变换还原
    repaired_final = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    return repaired_final