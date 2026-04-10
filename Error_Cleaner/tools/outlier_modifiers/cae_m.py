import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def cae_m_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None, 
    window: int = 10,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    latent_dim: int = 8,
    contamination: float = 0.05
) -> np.ndarray:
    """
    自适应 AE 修复函数（不依赖 label）：
    1. 自学习：假设大部分数据是正常的，模型学习数据的低维流形。
    2. 自检测：计算重构残差，残差过大的点被识别为“脏数据”。
    3. 协同修复：利用解码器的输出（最接近正常的模式）强制覆盖脏点和 NaN。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 鲁棒标准化 ---
    data_flat = data_abnormal.reshape(-1, n_features)
    medians = np.nanmedian(data_flat, axis=0)
    # 填充 NaN 以供训练
    data_prefilled = np.where(np.isnan(data_abnormal), medians, data_abnormal)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_prefilled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- 2. 构建训练集 (滑动窗口) ---
    X_train_list = []
    for s in range(n_samples):
        for i in range(n_timestamps - window + 1):
            X_train_list.append(data_scaled[s, i : i + window])

    X_train_tensor = torch.tensor(np.array(X_train_list))
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # --- 3. 定义模型 ---
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
                nn.Linear(128, input_dim) 
            )

        def forward(self, x):
            B, L, F = x.shape
            encoded = self.encoder(x.reshape(B, -1))
            decoded = self.decoder(encoded).reshape(B, L, F)
            return decoded

    model = MLPAE(window, n_features, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- 4. 训练：学习数据的通用表征 ---
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

    # --- 5. 自动识别与修复 ---
    
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            # 准备当前样本
            sample_tensor = torch.tensor(data_scaled[s]).unsqueeze(0) # (1, L, F)
            
            # 分段重构以平滑结果
            full_reconstruction = np.zeros_like(data_scaled[s])
            counts = np.zeros(n_timestamps)
            
            for t in range(n_timestamps - window + 1):
                win_tensor = torch.tensor(data_scaled[s, t : t + window]).unsqueeze(0).to(device)
                reconstructed_win = model(win_tensor).cpu().numpy()[0]
                full_reconstruction[t : t + window] += reconstructed_win
                counts[t : t + window] += 1
            
            full_reconstruction /= counts[:, np.newaxis]
            
            # 判定脏数据：计算重构误差
            residuals = np.mean((data_scaled[s] - full_reconstruction)**2, axis=1)
            # 动态阈值：基于分位数，排除掉重构效果最差的点
            threshold = np.percentile(residuals, 100 * (1 - contamination))
            
            # 修复掩码：误差过大的点 或 原始 NaN 点
            mask_is_nan = np.isnan(data_abnormal[s]).any(axis=1)
            mask_is_dirty = (residuals > threshold) | mask_is_nan
            
            # 使用自编码器生成的“理想值”覆盖脏点
            data_repaired_scaled[s, mask_is_dirty] = full_reconstruction[mask_is_dirty]

    # --- 6. 还原 ---
    repaired_final = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    return repaired_final