import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class CATCH_Model:
    def __init__(
        self,
        window_size: int = 64,
        patch_len: int = 8,          # 频率轴上每个 patch 包含的频点数
        d_model: int = 128,           # Transformer 的特征维度
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        stride: int = 1,             # 滑动窗口步长
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.window_size = window_size
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads if d_model % n_heads == 0 else 1
        
        self.model = CATCH_NN(
            d_model=d_model,
            n_heads=self.n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)
        
        self.is_fitted = False
        print(f"CATCH Model initialized on {self.device}")

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        """Convert (T, C) -> (N, window_size, C)"""
        T, C = X.shape
        if T < self.window_size:
            raise ValueError(f"Time series too short: {T} < {self.window_size}")
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(X[i:i + self.window_size])
        return np.stack(windows)  # (N, W, C)

    def _freq_patching(self, x_amp: torch.Tensor) -> torch.Tensor:
        """
        Input:  (N, F, C)  —— F = window_size // 2 + 1
        Output: (N, P, D)  —— D = patch_len * C
        """
        N, F, C = x_amp.shape
        max_patches = F // self.patch_len
        if max_patches == 0:
            raise ValueError(f"Frequency bins ({F}) too few for patch_len={self.patch_len}")
        trimmed = x_amp[:, :max_patches * self.patch_len, :]
        patches = trimmed.reshape(N, max_patches, self.patch_len * C)
        return patches

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, epochs: int = 200, lr: float = 1e-3):
        """
        Fit on a single multivariate time series.
        X: (n_timestamps, 1, n_features)  ← your required format
        y: (n_timestamps,)                ← ignored (for compatibility)
        """
        # Validate and preprocess input shape
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")
        
        X_squeezed = X.squeeze(1)  # (T, C)
        X_windows = self._sliding_window(X_squeezed)  # (N, W, C)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        # FFT → amplitude spectrum
        x_fft = torch.fft.rfft(X_tensor, dim=1)   # (N, F, C), F = W//2 + 1
        x_amp = torch.abs(x_fft)                  # (N, F, C)

        # Frequency patching
        patches = self._freq_patching(x_amp)      # (N, P, D)

        # Ensure patch dim <= d_model (pad if needed)
        D = patches.shape[-1]
        if D > self.d_model:
            patches = patches[..., :self.d_model]
        elif D < self.d_model:
            patches = F.pad(patches, (0, self.d_model - D))

        # Training loop
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = self.model(patches)
            loss = F.mse_loss(recon, patches)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        print(f"CATCH training finished. Final loss: {loss.item():.6f}")
        return self

    def decision_scores_(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute per-timestamp anomaly scores.
        Input:
            X: (n_timestamps, 1, n_features)
            y: (n_timestamps,) — ignored
        Output:
            scores: (n_timestamps,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")

        T = X.shape[0]
        scores_per_timestamp = np.zeros(T)

        if T < self.window_size:
            return scores_per_timestamp

        X_squeezed = X.squeeze(1)  # (T, C)
        X_windows = self._sliding_window(X_squeezed)  # (N, W, C)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        # FFT + patching
        x_fft = torch.fft.rfft(X_tensor, dim=1)
        x_amp = torch.abs(x_fft)
        patches = self._freq_patching(x_amp)

        # Pad to d_model
        D = patches.shape[-1]
        if D > self.d_model:
            patches = patches[..., :self.d_model]
        elif D < self.d_model:
            patches = F.pad(patches, (0, self.d_model - D))

        # Inference
        self.model.eval()
        with torch.no_grad():
            recon = self.model(patches)
            error = torch.mean((recon - patches) ** 2, dim=[1, 2])  # (N,)

        error = error.cpu().numpy()

        # Map window scores to timestamps (assign to last timestamp of each window)
        for i, score in enumerate(error):
            t_end = i * self.stride + self.window_size - 1
            if t_end < T:
                scores_per_timestamp[t_end] = score

        # Forward-fill leading zeros for robustness
        for i in range(1, T):
            if scores_per_timestamp[i] == 0 and scores_per_timestamp[i - 1] != 0:
                scores_per_timestamp[i] = scores_per_timestamp[i - 1]

        return scores_per_timestamp


class CATCH_NN(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, P, d_model)
        return self.encoder(x)