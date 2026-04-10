import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

def gan_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None,
    noise_dim=10,
    epochs=50,                 # GAN 通常需要更多迭代
    batch_size=64,
    contamination=0.05
):
    """
    自适应 GAN 修复函数（不依赖 label）：
    1. 自动判定：利用 MAD 统计量自动识别潜在脏数据并生成掩码。
    2. 鲁棒训练：生成器尝试填充脏点，判别器尝试区分“原始干净点”与“修复后的脏点”。
    3. 空间一致性：强制模型学习多变量之间的隐式相关性。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 数据识别与归一化 ---
    # 自动识别脏数据（离群点判定）
    data_flat = data_abnormal.reshape(-1, n_features)
    medians = np.nanmedian(data_flat, axis=0)
    mads = np.nanmedian(np.abs(data_flat - medians), axis=0)
    
    # 判定准则：NaN 或 偏离中位数 > 3倍MAD
    # 防止 mads 为 0
    mads = np.where(mads == 0, 1e-6, mads)
    is_dirty_flat = np.isnan(data_flat).any(axis=1) | \
                    np.any(np.abs(data_flat - medians) > 3 * mads, axis=1)
    
    # 初步填补用于拟合 Scaler
    data_prefilled = np.where(np.isnan(data_flat), medians, data_flat)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 仅用判定为“干净”的点来 fit
    clean_data = data_prefilled[~is_dirty_flat]
    if len(clean_data) < 10: # 极端情况
        scaler.fit(data_prefilled)
    else:
        scaler.fit(clean_data)
        
    data_scaled = scaler.transform(data_prefilled).reshape(n_samples, n_timestamps, n_features)
    is_dirty_3d = is_dirty_flat.reshape(n_samples, n_timestamps)

    # --- 2. 准备训练集 ---
    # 仅提取被判定为干净的观测点作为“真值”样本
    X_real_train = data_scaled[~is_dirty_3d].astype("float32")
    if len(X_real_train) == 0:
        return data_abnormal
    X_real_tensor = torch.from_numpy(X_real_train).to(device)

    # --- 3. 定义模型 ---
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, n_features),
                nn.Tanh()
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
    
    opt_g = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.9))
    opt_d = optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.9))
    criterion = nn.BCEWithLogitsLoss()

    # --- 4. 训练过程 ---
    gen.train(); disc.train()
    for epoch in range(epochs):
        # 判别器训练
        idx = torch.randint(0, X_real_tensor.shape[0], (batch_size,))
        real_data = X_real_tensor[idx]
        
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_data = gen(noise)
        
        # 标签平滑提升稳定性
        real_labels = torch.full((batch_size, 1), 0.9).to(device)
        fake_labels = torch.full((batch_size, 1), 0.1).to(device)
        
        opt_d.zero_grad()
        d_loss_real = criterion(disc(real_data), real_labels)
        d_loss_fake = criterion(disc(fake_data.detach()), fake_labels)
        (d_loss_real + d_loss_fake).backward()
        opt_d.step()
        
        # 生成器训练
        opt_g.zero_grad()
        g_loss = criterion(disc(fake_data), torch.ones(batch_size, 1).to(device))
        g_loss.backward()
        opt_g.step()

    # --- 5. 自动修复 ---
    gen.eval()
    data_repaired_scaled = data_scaled.copy()
    with torch.no_grad():
        for s in range(n_samples):
            dirty_idx = is_dirty_3d[s]
            n_dirty = np.sum(dirty_idx)
            if n_dirty > 0:
                # 提示：为了让修复更符合当前样本，可以使用当前样本的均值作为噪声基准
                # 这里保持随机噪声以获得生成的“多样性”
                z = torch.randn(n_dirty, noise_dim).to(device)
                repaired_points = gen(z).cpu().numpy()
                data_repaired_scaled[s, dirty_idx] = repaired_points

    # 逆变换还原
    final_data = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    return final_data