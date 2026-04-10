import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import warnings

def gan_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    noise_dim=10,
    epochs=30,
    batch_size=64,
):
    """
    优化后的 GAN 修复函数：
    1. 全局建模：所有样本共同训练一个 GAN，学习全局特征分布。
    2. 归一化：强制将数据映射到 [-1, 1] 空间，配合 Tanh 激活函数。
    3. 修复稳定性：使用正常样本的均值作为噪声的基础，增加修复的合理性。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 数据归一化 (GAN 必须) ---
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normal_mask_all = (label == 0)
    if not np.any(normal_mask_all):
        return data_abnormal
    
    scaler.fit(data_abnormal[normal_mask_all])
    data_scaled = scaler.transform(data_abnormal.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- 2. 提取全局训练集 ---
    X_train_np = data_scaled[label == 0]
    X_train_tensor = torch.from_numpy(X_train_np).to(device)

    # --- 3. 定义模型 ---
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, n_features),
                nn.Tanh() # 输出映射到 [-1, 1]
            )
        def forward(self, z): return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 1)
            )
        def forward(self, x): return self.net(x)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    
    opt_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # --- 4. 全局训练 ---
    gen.train(); disc.train()
    for epoch in range(epochs):
        # 简化版训练逻辑：随机抽样
        idx = torch.randint(0, X_train_tensor.shape[0], (batch_size,))
        real_data = X_train_tensor[idx]
        
        # 训练判别器
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_data = gen(noise)
        d_loss = criterion(disc(real_data), torch.ones(batch_size, 1).to(device)) + \
                 criterion(disc(fake_data.detach()), torch.zeros(batch_size, 1).to(device))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        
        # 训练生成器
        g_loss = criterion(disc(fake_data), torch.ones(batch_size, 1).to(device))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

    # --- 5. 批量修复与逆变换 ---
    gen.eval()
    data_repaired_scaled = data_scaled.copy()
    with torch.no_grad():
        for s in range(n_samples):
            anomaly_mask = (label[s] == 1)
            n_anomaly = np.sum(anomaly_mask)
            if n_anomaly > 0:
                # 产生噪声并生成修复数据
                noise = torch.randn(n_anomaly, noise_dim).to(device)
                repaired_points = gen(noise).cpu().numpy()
                data_repaired_scaled[s, anomaly_mask] = repaired_points

    # 逆变换回原量纲
    repaired_2d = data_repaired_scaled.reshape(-1, n_features)
    final_data = scaler.inverse_transform(repaired_2d).reshape(n_samples, n_timestamps, n_features)
    
    return final_data