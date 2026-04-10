import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# 官方 GitHub: https://github.com/daidahao/SARAD/
# Paper: NeurIPS 2024, "Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series"

class SARAD_Model:
    def __init__(
        self,
        window_size: int = 64,
        d_model: int = 64,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        stride: int = 1,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.window_size = window_size
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads if d_model % n_heads == 0 else 1
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 模型将在 fit 时根据实际 n_features 构建
        self.model = None
        self.is_fitted = False
        print(f"SARAD Model initialized on {self.device}")

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        """Convert (T, C) -> (N, W, C)"""
        T, C = X.shape
        if T < self.window_size:
            raise ValueError(f"Time series too short: {T} < {self.window_size}")
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(X[i:i + self.window_size])
        return np.stack(windows)  # (N, W, C)

    def _compute_association_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial association matrix via cosine similarity.
        Input:  (N, W, C)
        Output: (N, C, C) — symmetric association matrix per window
        """
        # Normalize over time dimension (W)
        X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)  # (N, W, C)
        # Cosine similarity: A = X^T X
        A = torch.bmm(X_norm.transpose(1, 2), X_norm)  # (N, C, C)
        return A

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, epochs: int = 200, lr: float = 1e-3):
        """
        Fit SARAD on normal data only.
        X: (n_timestamps, 1, n_features)
        y: (n_timestamps,) — ignored (for compatibility)
        """
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")
        
        X_squeezed = X.squeeze(1)  # (T, C)
        X_windows = self._sliding_window(X_squeezed)  # (N, W, C)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        # Step 1: Compute spatial association matrices
        A = self._compute_association_matrix(X_tensor)  # (N, C, C)
        N, C, _ = A.shape

        # Step 2: Build model if not exists (now we know C)
        if self.model is None:
            self.model = SARAD_NN(
                n_features=C,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout
            ).to(self.device)

        # Step 3: Train autoencoder in association space
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        A_flat = A.view(N, -1)  # (N, C*C)

        for epoch in range(epochs):
            optimizer.zero_grad()
            recon_flat = self.model(A_flat)
            loss = nn.MSELoss()(recon_flat, A_flat)
            loss.backward()
            optimizer.step()

        self.is_fitted = True
        print(f"SARAD training finished. Final loss: {loss.item():.6f}")
        return self

    def decision_scores_(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return anomaly scores per timestamp.
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

        X_squeezed = X.squeeze(1)
        X_windows = self._sliding_window(X_squeezed)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        A = self._compute_association_matrix(X_tensor)  # (N, C, C)
        N = A.shape[0]
        A_flat = A.view(N, -1)

        self.model.eval()
        with torch.no_grad():
            recon_flat = self.model(A_flat)
            error = torch.mean((recon_flat - A_flat) ** 2, dim=1)  # (N,)

        error = error.cpu().numpy()

        # Assign score to last timestamp of each window
        for i, score in enumerate(error):
            t_end = i * self.stride + self.window_size - 1
            if t_end < T:
                scores_per_timestamp[t_end] = score

        # Forward-fill for robustness
        for i in range(1, T):
            if scores_per_timestamp[i] == 0 and scores_per_timestamp[i - 1] != 0:
                scores_per_timestamp[i] = scores_per_timestamp[i - 1]

        return scores_per_timestamp


class SARAD_NN(nn.Module):
    """
    Autoencoder in association space.
    Input: flattened association matrix (C*C,)
    Output: reconstructed flattened association matrix
    Uses Transformer to capture inter-feature dependencies.
    """
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        input_dim = n_features * n_features

        # Project to d_model
        self.proj_in = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))  # dummy positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C*C)
        x = self.proj_in(x)  # (N, d_model)
        x = x.unsqueeze(1) + self.pos_encoding  # (N, 1, d_model)
        x = self.transformer(x)  # (N, 1, d_model)
        x = x.squeeze(1)  # (N, d_model)
        x = self.proj_out(x)  # (N, C*C)
        return x