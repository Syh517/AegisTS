import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def imdiffusion_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None,
    timesteps=50,
    epochs=15,
    batch_size=64,
    lr=0.001,
    contamination=0.05
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 鲁棒标准化 ---
    data_float = data_abnormal.astype(np.float64, copy=True)
    invalid_mask = ~np.isfinite(data_float)
    data_float[invalid_mask] = np.nan

    data_flat = data_float.reshape(-1, n_features)
    medians = np.nanmedian(data_flat, axis=0)
    # 某些特征可能全是无效值，回退到 0，避免填充后仍含 NaN
    medians = np.where(np.isfinite(medians), medians, 0.0)
    # 填充 NaN/Inf 等无效值供模型训练
    data_prefilled = np.where(np.isnan(data_float), medians, data_float)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_prefilled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- 2. 扩散调度器 (DDPM) ---
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- 3. 准备自监督训练集 ---
    # 在无 label 情况下，我们假设大部分点是正常的，训练模型学习全局分布
    X_train_tensor = torch.from_numpy(data_scaled.reshape(-1, n_features)).to(device)
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # --- 4. 定义 DiffusionNet (带有时间步 Embedding) ---
    class DiffusionNet(nn.Module):
        def __init__(self, feat_dim):
            super().__init__()
            self.time_mlp = nn.Sequential(
                nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32)
            )
            self.net = nn.Sequential(
                nn.Linear(feat_dim + 32, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, feat_dim)
            )
        def forward(self, x, t):
            t_emb = self.time_mlp(t.float().view(-1, 1) / timesteps)
            return self.net(torch.cat([x, t_emb], dim=1))

    model = DiffusionNet(n_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- 5. 训练：学习逆转噪声 ---
    model.train()
    for _ in range(epochs):
        for (x_0,) in loader:
            t = torch.randint(0, timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            x_t = sqrt_alphas_cumprod[t].view(-1, 1) * x_0 + \
                  sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
            
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x_t, t), noise)
            loss.backward()
            optimizer.step()

    # --- 6. 自动检测与修复 (Inpainting 逻辑) ---
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            x_orig = torch.from_numpy(data_scaled[s]).to(device)
            
            # A. 自动识别：利用 T=1 时的重构误差作为异常分数
            # 正常数据在低噪声下重构极准，脏数据则不然
            t_small = torch.zeros(n_timestamps, device=device).long()
            noise_pred = model(x_orig, t_small)
            # 这里的误差反映了点是否符合训练出的“标准分布”
            error_score = torch.mean(noise_pred**2, dim=1).cpu().numpy()
            
            # 合并 NaN 掩码与高误差掩码
            mask_is_invalid = ~np.isfinite(data_abnormal[s]).all(axis=1)
            thresh = np.percentile(error_score, 100 * (1 - contamination))
            m = torch.from_numpy((error_score > thresh) | mask_is_invalid).view(-1, 1).to(device).float()

            # B. 扩散采样修复 (Guided Inpainting)
            x_curr = torch.randn(n_timestamps, n_features, device=device)
            for t in reversed(range(timesteps)):
                t_tensor = torch.full((n_timestamps,), t, device=device)
                eps_theta = model(x_curr, t_tensor)
                
                # 基础去噪步
                alpha_t = alphas[t]
                alpha_t_bar = alphas_cumprod[t]
                coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)
                x_prev = (1 / torch.sqrt(alpha_t)) * (x_curr - coeff * eps_theta)
                
                if t > 0:
                    x_prev += torch.sqrt(betas[t]) * torch.randn_like(x_curr)
                
                # 关键：掩码引导。保留识别出的正常点，仅让模型生成脏点
                # 给原始正常点加上对应当前步的噪声，保持与采样过程的分布一致
                x_orig_noisy = sqrt_alphas_cumprod[t] * x_orig + sqrt_one_minus_alphas_cumprod[t] * torch.randn_like(x_orig)
                x_curr = (1 - m) * x_orig_noisy + m * x_prev
            
            data_repaired_scaled[s] = x_curr.cpu().numpy()

    # --- 7. 还原 ---
    repaired_reshaped = data_repaired_scaled.reshape(-1, n_features)
    final_data = scaler.inverse_transform(repaired_reshaped).reshape(n_samples, n_timestamps, n_features)
    return final_data