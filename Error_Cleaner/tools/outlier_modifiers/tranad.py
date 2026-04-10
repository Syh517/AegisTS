import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def tranad_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None,
    window=10,
    epochs=15,
    batch_size=64,
    lr=0.001,
    contamination=0.05
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- 1. 鲁棒标准化 ---
    # 使用中位数填补 NaN，防止异常值拉偏 scaler
    data_flat = data_abnormal.reshape(-1, n_features)
    medians = np.nanmedian(data_flat, axis=0)
    data_prefilled = np.where(np.isnan(data_abnormal), medians, data_abnormal)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_prefilled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- 2. 构建 Transformer 训练集 ---
    # 在无 label 情况下，让模型学习重构整个序列
    X_train_list = []
    for s in range(n_samples):
        for i in range(n_timestamps - window + 1):
            X_train_list.append(data_scaled[s, i : i + window])

    X_train_tensor = torch.tensor(np.array(X_train_list)).to(device)
    loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)

    # --- 3. 定义 Transformer 模型 ---
    class TransformerReconstructor(nn.Module):
        def __init__(self, F, d_model=64, nhead=4):
            super().__init__()
            self.input_fc = nn.Linear(F, d_model)
            # 简单的 Transformer 编码器层
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_fc = nn.Linear(d_model, F)

        def forward(self, x):
            x = self.input_fc(x)
            x = self.transformer(x)
            return self.output_fc(x)

    model = TransformerReconstructor(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- 4. 自监督训练 ---
    model.train()
    for epoch in range(epochs):
        for xb, in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, xb)
            loss.backward()
            optimizer.step()

    # --- 5. 自适应检测与修复 ---
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            sample_tensor = torch.tensor(data_scaled[s]).unsqueeze(0).to(device)
            # 全量重构
            # 对于长序列，Transformer 可以直接处理，或者采用滑动窗口求平均
            reconstructed = model(sample_tensor).cpu().numpy()[0]

            # 计算点对点的重构误差
            errors = np.mean((data_scaled[s] - reconstructed)**2, axis=1)
            # 根据污染率动态设置异常阈值
            threshold = np.percentile(errors, 100 * (1 - contamination))
            
            # 定义脏数据掩码：NaN 点 或 误差极大的离群点
            mask_is_nan = np.isnan(data_abnormal[s]).any(axis=1)
            mask_to_repair = (errors > threshold) | mask_is_nan
            
            # 使用 Transformer 的重构值进行覆盖修复
            data_repaired_scaled[s, mask_to_repair] = reconstructed[mask_to_repair]

    # --- 6. 逆变换还原 ---
    repaired_final = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    
    return repaired_final