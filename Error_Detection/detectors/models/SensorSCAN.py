import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import Optional

class SensorSCAN_Model:
    def __init__(
        self,
        window_size: int = 64,
        stride: int = 1,
        n_clusters: Optional[int] = None,
        hidden_dim: int = 128,
        temperature: float = 0.1,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.window_size = window_size
        self.stride = stride
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        self.encoder = None
        self.decoder = None
        self.cluster_centers = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_fitted = False
        print(f"SensorSCAN Model initialized on {self.device}")

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        T, C = X.shape
        if T < self.window_size:
            raise ValueError(f"Time series too short: {T} < {self.window_size}")
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(X[i:i + self.window_size])
        return np.stack(windows)

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.scaler_mean = X.mean(axis=0, keepdims=True)
        self.scaler_std = X.std(axis=0, keepdims=True)
        self.scaler_std[self.scaler_std == 0] = 1.0
        return (X - self.scaler_mean) / self.scaler_std

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.scaler_mean) / self.scaler_std

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        ssl_epochs: int = 200,
        scan_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32
    ) -> 'SensorSCAN_Model':
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")
        
        X_squeezed = X.squeeze(1)
        X_norm = self._standardize_fit(X_squeezed)
        X_windows = self._sliding_window(X_norm)
        N, W, C = X_windows.shape

        if self.n_clusters is None:
            self.n_clusters = min(10, max(2, N // 100))

        # Build autoencoder
        self.encoder = SensorSCAN_Encoder(C, W, self.hidden_dim).to(self.device)
        self.decoder = SensorSCAN_Decoder(C, W, self.hidden_dim).to(self.device)

        X_tensor = torch.from_numpy(X_windows).float().to(self.device)
        X_tensor = X_tensor.permute(0, 2, 1)  # (N, C, W)

        # === Stage 1: Autoencoder Pretraining (Stable!) ===
        print("Stage 1: Autoencoder Pretraining...")
        self._ae_pretrain(X_tensor, epochs=ssl_epochs, lr=lr, batch_size=batch_size)

        # === Stage 2: SCAN Clustering ===
        print("Stage 2: SCAN Clustering...")
        with torch.no_grad():
            Z = self.encoder(X_tensor)
            Z_np = Z.cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(Z_np)
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        self._scan_finetune(X_tensor, epochs=scan_epochs, lr=lr, batch_size=batch_size)

        self.is_fitted = True
        print("SensorSCAN training finished.")
        return self

    def _ae_pretrain(self, X: torch.Tensor, epochs: int, lr: float, batch_size: int):
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.encoder.train()
        self.decoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(self.device)
                z = self.encoder(batch)
                recon = self.decoder(z)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 100 == 0:
                print(f"  AE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    def _scan_finetune(self, X: torch.Tensor, epochs: int, lr: float, batch_size: int):
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        self.encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(self.device)
                z = self.encoder(batch)
                distances = torch.cdist(z, self.cluster_centers)  # (B, K)
                q = F.softmax(-distances / self.temperature, dim=1)
                # Target distribution p
                p = q ** 2 / (q.sum(dim=0, keepdim=True) + 1e-8)
                p = (p.T / (p.sum(dim=1) + 1e-8)).T
                loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 100 == 0:
                print(f"  SCAN Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    def decision_scores_(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")

        T = X.shape[0]
        scores = np.zeros(T)
        if T < self.window_size:
            return scores

        X_squeezed = X.squeeze(1)
        X_norm = self._standardize_transform(X_squeezed)
        X_windows = self._sliding_window(X_norm)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)
        X_tensor = X_tensor.permute(0, 2, 1)

        self.encoder.eval()
        with torch.no_grad():
            Z = self.encoder(X_tensor)
            distances = torch.cdist(Z, self.cluster_centers)
            window_scores = distances.min(dim=1).values.cpu().numpy()

        for i, score in enumerate(window_scores):
            t_end = i * self.stride + self.window_size - 1
            if t_end < T:
                scores[t_end] = score

        for i in range(1, T):
            if scores[i] == 0 and scores[i - 1] != 0:
                scores[i] = scores[i - 1]

        return scores


class SensorSCAN_Encoder(nn.Module):
    def __init__(self, input_channels: int, window_size: int, hidden_dim: int = 128):
        super().__init__()
        k1 = min(5, window_size)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=k1)
        out_len1 = window_size - k1 + 1
        k2 = min(3, out_len1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=k2)
        out_len2 = out_len1 - k2 + 1
        self.fc = nn.Linear(128 * out_len2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SensorSCAN_Decoder(nn.Module):
    def __init__(self, input_channels: int, window_size: int, hidden_dim: int = 128):
        super().__init__()
        k1 = min(5, window_size)
        out_len1 = window_size - k1 + 1
        k2 = min(3, out_len1)
        out_len2 = out_len1 - k2 + 1
        self.fc = nn.Linear(hidden_dim, 128 * out_len2)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=k2)
        self.deconv1 = nn.ConvTranspose1d(64, input_channels, kernel_size=k1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, -1)
        x = F.relu(self.deconv2(x))
        x = self.deconv1(x)
        return x