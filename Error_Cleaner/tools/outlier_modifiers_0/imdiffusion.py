import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings

def imdiffusion_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray,
    timesteps=50,
    epochs=15,
    batch_size=64,
    lr=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # 1. 全局标准化 (Diffusion 对数值稳定性极其敏感)
    scaler = StandardScaler()
    normal_mask_all = (label == 0)
    if not np.any(normal_mask_all): return data_abnormal
    
    scaler.fit(data_abnormal[normal_mask_all])
    data_scaled = scaler.transform(data_abnormal.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # 2. 预计算扩散调度器参数 (调度器在 GPU 上预存)
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # 3. 提取全局训练样本 (只用正常点作为 X_0)
    X_train_raw = data_scaled[label == 0]
    X_train_tensor = torch.from_numpy(X_train_raw).to(device)
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # 4. 定义模型 (带时间嵌入的 MLP)
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
    criterion = nn.MSELoss()

    # 5. 全局训练 (预测噪声 epsilon)
    model.train()
    for _ in range(epochs):
        for (x_0,) in loader:
            t = torch.randint(0, timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            # 重参数化技巧加噪
            x_t = sqrt_alphas_cumprod[t].view(-1, 1) * x_0 + \
                  sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
            
            optimizer.zero_grad()
            loss = criterion(model(x_t, t), noise)
            loss.backward()
            optimizer.step()

    # 6. 反向采样修复 (核心：掩码引导)
    # 
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            mask_anomaly = (label[s] == 1)
            if not np.any(mask_anomaly): continue
            
            # 初始状态：对于整个 sample 进行修复，但保留已知正常点
            x_curr = torch.randn(n_timestamps, n_features, device=device)
            x_orig = torch.from_numpy(data_scaled[s]).to(device)
            m = torch.from_numpy(label[s]).view(-1, 1).to(device) # 1 为异常

            for t in reversed(range(timesteps)):
                t_tensor = torch.full((n_timestamps,), t, device=device)
                eps_theta = model(x_curr, t_tensor)
                
                # 基础去噪步骤 (DDPM 采样)
                alpha_t = alphas[t]
                alpha_t_bar = alphas_cumprod[t]
                coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)
                x_prev = (1 / torch.sqrt(alpha_t)) * (x_curr - coeff * eps_theta)
                
                if t > 0:
                    noise = torch.randn_like(x_curr)
                    x_prev += torch.sqrt(betas[t]) * noise
                
                # --- 条件引导 (Conditioning) ---
                # 正常点：用原值加噪声（对应当前步的噪声水平）
                # 异常点：使用模型生成的预测值
                x_orig_t = sqrt_alphas_cumprod[t] * x_orig + sqrt_one_minus_alphas_cumprod[t] * torch.randn_like(x_orig)
                x_curr = (1 - m) * x_orig_t + m * x_prev
            
            data_repaired_scaled[s] = x_curr.cpu().numpy()

    # 7. 逆变换还原
    repaired_final = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    return repaired_final