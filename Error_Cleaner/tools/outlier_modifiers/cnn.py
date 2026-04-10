import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def cnn_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None, 
    window=16,          # 卷积窗口通常设为 2 的幂次方
    epochs=15,
    batch_size=32,
    contamination=0.05  # 预估的数据污染比例
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 鲁棒标准化 ---
    # 使用中位数填补缺失值作为训练底色
    flat_data = data_abnormal.reshape(-1, n_features)
    medians = np.nanmedian(flat_data, axis=0)
    data_prefilled = np.where(np.isnan(data_abnormal), medians, data_abnormal)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_prefilled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- 2. 构造 1D-CNN 数据集 (Batch, Channel, Length) ---
    # CNN 的输入格式通常要求频道（特征）在中间
    X_train = []
    for s in range(n_samples):
        for i in range(n_timestamps - window + 1):
            X_train.append(data_scaled[s, i : i + window].T) # 转置为 (n_features, window)

    X_train_tensor = torch.tensor(np.array(X_train))

    # --- 3. 定义 1D-CNN 自编码器 ---
    class CNNAutoEncoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            # Encoder: 提取空间和局部时间特征
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 16, kernel_size=3, padding=1),
                nn.ReLU()
            )
            # Decoder: 还原信号
            self.decoder = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, n_features, kernel_size=3, padding=1)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = CNNAutoEncoder(n_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    # --- 4. 快速并行训练 ---
    model.train()
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for xb, in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, xb)
            loss.backward()
            optimizer.step()

    # --- 5. 自动检测与覆盖 ---
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            # 准备样本数据 (1, n_features, n_timestamps)
            sample_input = torch.tensor(data_scaled[s].T).unsqueeze(0).to(device)
            
            # 为了处理非 window 倍数的序列，直接全量填充
            # CNN 可以处理任意长度输入，只要 padding 设置正确
            reconstructed_full = model(sample_input).cpu().numpy()[0].T # 转回 (n_timestamps, n_features)

            # 计算每个点的重构误差
            errors = np.mean((data_scaled[s] - reconstructed_full)**2, axis=1)
            threshold = np.percentile(errors, 100 * (1 - contamination))
            
            # 掩码：原 NaN 点 或 误差过大的脏点
            mask_is_nan = np.isnan(data_abnormal[s]).any(axis=1)
            mask_to_fix = (errors > threshold) | mask_is_nan
            
            # 修复执行
            data_repaired_scaled[s, mask_to_fix] = reconstructed_full[mask_to_fix]

    # --- 6. 逆变换还原 ---
    final_data = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    return final_data