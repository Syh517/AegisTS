import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

def lstm_sequence_repair(
    data_abnormal: np.ndarray,
    label: np.ndarray = None, 
    window=10,
    epochs=20,
    batch_size=64,
    hidden_dim=64,
    contamination=0.05
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_timestamps, n_features = data_abnormal.shape
    
    # --- Step 1: 鲁棒归一化 ---
    # 既然不知道谁是脏数据，先用中位数填补 NaN，再进行标准化
    data_flat = data_abnormal.reshape(-1, n_features)
    # 使用 nanmedian 避免脏数据拉偏统计量
    medians = np.nanmedian(data_flat, axis=0)
    
    # 临时填补用于训练
    data_filled = np.where(np.isnan(data_abnormal), medians, data_abnormal)
    
    scaler = StandardScaler()
    # 全量拟合（在没有 label 时，这是最稳妥的起手式）
    data_scaled = scaler.fit_transform(data_filled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)
    data_scaled = data_scaled.astype("float32")

    # --- Step 2: 构造自监督训练集 (滑动窗口重构) ---
    # 目标：让模型学习“如何重构一个窗口”，正常点重构得好，脏点重构得差
    X_train_list = []
    for s in range(n_samples):
        for i in range(n_timestamps - window + 1):
            X_train_list.append(data_scaled[s, i : i + window])

    X_train_tensor = torch.tensor(np.array(X_train_list), dtype=torch.float32)

    # --- Step 3: 模型定义 (LSTM Autoencoder) ---
    class LSTMAutoEncoder(nn.Module):
        def __init__(self, input_dim, hidden):
            super().__init__()
            # Encoder
            self.encoder = nn.LSTM(input_dim, hidden, batch_first=True)
            # Decoder
            self.decoder = nn.LSTM(hidden, input_dim, batch_first=True)
            self.fc = nn.Linear(input_dim, input_dim)

        def forward(self, x):
            # x: (batch, window, features)
            _, (h, _) = self.encoder(x) # h: (1, batch, hidden)
            # 将隐变量平铺到每个时间步
            h_repeated = h.transpose(0, 1).repeat(1, window, 1)
            out, _ = self.decoder(h_repeated)
            return self.fc(out)

    model = LSTMAutoEncoder(n_features, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # --- Step 4: 训练 (学习通用时序模式) ---
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

    # --- Step 5: 检测与修复 ---
    model.eval()
    data_repaired_scaled = data_scaled.copy()
    
    with torch.no_grad():
        for s in range(n_samples):
            sample_tensor = torch.tensor(data_scaled[s]).unsqueeze(0).to(device)
            # 因为序列可能很长，我们需要分段重构
            reconstructed_sample = np.zeros_like(data_scaled[s])
            
            # 使用滑动窗口进行推理，取平均重构值
            count_map = np.zeros(n_timestamps)
            for i in range(n_timestamps - window + 1):
                window_data = sample_tensor[:, i:i+window, :]
                pred_window = model(window_data).cpu().numpy()[0]
                reconstructed_sample[i:i+window, :] += pred_window
                count_map[i:i+window] += 1
            
            reconstructed_sample /= count_map[:, np.newaxis]

            # 判定哪些点是“脏数据”：重构误差 > 阈值
            errors = np.mean((data_scaled[s] - reconstructed_sample)**2, axis=1)
            # 使用分位数作为动态阈值
            threshold = np.percentile(errors, 100 * (1 - contamination))
            
            mask_is_nan = np.isnan(data_abnormal[s]).any(axis=1)
            mask_is_dirty = (errors > threshold) | mask_is_nan
            
            # 修复：用重构值替换脏点
            data_repaired_scaled[s, mask_is_dirty] = reconstructed_sample[mask_is_dirty]

    # --- Step 6: 逆变换 ---
    final_data = scaler.inverse_transform(data_repaired_scaled.reshape(-1, n_features)).reshape(n_samples, n_timestamps, n_features)

    return final_data