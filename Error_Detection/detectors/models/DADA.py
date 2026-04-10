import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class DADA_Model:
    def __init__(
        self,
        window_size: int = 64,
        stride: int = 1,
        hidden_dim: int = 64,
        bottleneck_dim: int = 32,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.window_size = window_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        self.encoder = None
        self.decoder_normal = None
        self.decoder_abnormal = None
        
        self.scaler_mean = None
        self.scaler_std = None
        self.is_fitted = False
        print(f"DADA Model initialized on {self.device}")

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        T, C = X.shape
        if T < self.window_size:
            return np.zeros((0, self.window_size, C))  # handle short series
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
        return (X - self.scaler_mean) / self.scaler_std

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        lambda_adv: float = 0.1,      # small weight!
        margin: float = 0.5           # hinge margin
    ) -> 'DADA_Model':
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")
        
        X_squeezed = X.squeeze(1)
        X_norm = self._standardize_fit(X_squeezed)
        X_windows = self._sliding_window(X_norm)
        if len(X_windows) == 0:
            raise ValueError("Time series too short for windowing.")
        N, W, C = X_windows.shape

        # Build model
        self.encoder = nn.GRU(C, self.hidden_dim, batch_first=True).to(self.device)
        self.decoder_normal = self._build_decoder(W, C).to(self.device)
        self.decoder_abnormal = self._build_decoder(W, C).to(self.device)

        X_tensor = torch.from_numpy(X_windows).float().to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder_normal.parameters()) +
            list(self.decoder_abnormal.parameters()),
            lr=lr,
            weight_decay=1e-5  # L2 regularization
        )

        # Gradient clipping
        max_grad_norm = 1.0

        print("DADA: Starting stable dual-decoder training...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(self.device)

                # Encode
                _, h_n = self.encoder(batch)
                h_n = h_n.squeeze(0)  # (B, hidden_dim)

                # Decode
                recon_n = self.decoder_normal(h_n)
                recon_a = self.decoder_abnormal(h_n)

                # Losses
                loss_n = F.mse_loss(recon_n, batch, reduction='mean')
                loss_a = F.mse_loss(recon_a, batch, reduction='mean')

                # Hinge-style adversarial loss:
                # We want loss_a to be LARGE, but bounded.
                # So we minimize: loss_n + λ * max(0, margin - loss_a)
                adv_term = torch.clamp(margin - loss_a, min=0.0)
                loss = loss_n + lambda_adv * adv_term

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.decoder_normal.parameters()) +
                    list(self.decoder_abnormal.parameters()),
                    max_grad_norm
                )
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    avg_loss_n = F.mse_loss(recon_n, batch).item()
                    avg_loss_a = F.mse_loss(recon_a, batch).item()
                print(f"  Epoch {epoch+1}/{epochs}, Total Loss: {total_loss/len(dataloader):.6f}, "
                      f"Loss_n: {avg_loss_n:.6f}, Loss_a: {avg_loss_a:.6f}")

        self.is_fitted = True
        print("DADA: Training finished.")
        return self

    def _build_decoder(self, window_size: int, n_features: int):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, window_size * n_features),
            nn.Tanh(),  
            nn.Unflatten(1, (window_size, n_features))
        )

    def decision_scores_(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")

        T = X.shape[0]
        scores = np.full(T, np.nan)  # use NaN for unassigned
        if T < self.window_size:
            return np.zeros(T)

        X_squeezed = X.squeeze(1)
        X_norm = self._standardize_transform(X_squeezed)
        X_windows = self._sliding_window(X_norm)
        if len(X_windows) == 0:
            return np.zeros(T)

        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        self.encoder.eval()
        self.decoder_normal.eval()
        self.decoder_abnormal.eval()

        with torch.no_grad():
            _, h_n = self.encoder(X_tensor)
            h_n = h_n.squeeze(0)
            recon_n = self.decoder_normal(h_n)
            recon_a = self.decoder_abnormal(h_n)

            loss_n = torch.mean((X_tensor - recon_n) ** 2, dim=[1, 2])  # (N,)
            loss_a = torch.mean((X_tensor - recon_a) ** 2, dim=[1, 2])  # (N,)
            window_scores = (loss_n - loss_a).cpu().numpy()  # DADA score

        # Assign to last timestamp of each window
        for i, score in enumerate(window_scores):
            t_end = i * self.stride + self.window_size - 1
            if t_end < T:
                scores[t_end] = score

        # Forward-fill NaNs
        last_valid = 0.0
        for i in range(T):
            if not np.isnan(scores[i]):
                last_valid = scores[i]
            else:
                scores[i] = last_valid

        return scores