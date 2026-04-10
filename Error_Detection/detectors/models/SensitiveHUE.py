import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Optional
import warnings

# 官方 GitHub: https://github.com/yuesuoqingqiu/SensitiveHUE

class SensitiveHUE_Model:
    def __init__(
        self,
        window_size: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = None  # Will be built in fit()
        self.is_fitted = False
        print(f"SensitiveHUE Model initialized on {self.device}")

    def _sliding_window(self, X: np.ndarray) -> np.ndarray:
        """Convert (T, C) -> (N, W, C)"""
        T, C = X.shape
        if T < self.window_size:
            raise ValueError(f"Time series too short: {T} < {self.window_size}")
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(X[i:i + self.window_size])
        return np.stack(windows)  # (N, W, C)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32
    ) -> 'SensitiveHUE_Model':
        """
        Train SensitiveHUE using MTS-NLL loss on normal data.
        X: (n_timestamps, 1, n_features)
        y: (n_timestamps,) — ignored (for compatibility)
        """
        if X.ndim != 3 or X.shape[1] != 1:
            raise ValueError(f"Expected X shape (T, 1, C), got {X.shape}")
        
        X_squeezed = X.squeeze(1)  # (T, C)
        
        # Check for NaN or infinite values in input data
        if np.isnan(X_squeezed).any() or np.isinf(X_squeezed).any():
            warnings.warn("Input data contains NaN or infinite values. Attempting to handle...")
            # Replace NaN with 0 and clip infinite values
            X_squeezed = np.nan_to_num(X_squeezed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize data to prevent numerical instability
        mean = np.mean(X_squeezed, axis=0, keepdims=True)
        std = np.std(X_squeezed, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        X_normalized = (X_squeezed - mean) / std
        
        X_windows = self._sliding_window(X_normalized)  # (N, W, C)
        N, W, C = X_windows.shape

        # Build model now that we know C
        if self.model is None:
            self.model = SensitiveHUE_NN(
                input_dim=C,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        # Prepare DataLoader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_windows).float())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Add gradient clipping to prevent exploding gradients
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch, in dataloader:
                batch = batch.to(self.device)  # (B, W, C)
                optimizer.zero_grad()
                
                mu, sigma = self.model(batch)  # (B, W, C), (B, W, C)
                
                # Check for NaN in model outputs
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    warnings.warn("Model produced NaN outputs. Skipping batch...")
                    continue
                
                nll = self._mts_nll_loss(batch, mu, sigma)
                
                # Check for valid loss
                if torch.isnan(nll) or torch.isinf(nll):
                    warnings.warn(f"Invalid loss detected: {nll}. Skipping batch...")
                    continue
                    
                nll.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += nll.item()
            
            avg_loss = total_loss/len(dataloader) if len(dataloader) > 0 else float('inf')
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Avg NLL Loss: {avg_loss:.4f}")

        self.is_fitted = True
        print("SensitiveHUE training finished.")
        return self

    def _mts_nll_loss(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """MTS-NLL Loss as described in the paper."""
        # Add small epsilon to avoid log(0) and ensure numerical stability
        sigma = torch.clamp(sigma, min=1e-6, max=1e6)
        # Clamp values to prevent overflow
        x = torch.clamp(x, -1e6, 1e6)
        mu = torch.clamp(mu, -1e6, 1e6)
        
        nll = torch.log(sigma) + (x - mu) ** 2 / (2 * sigma ** 2)
        # Return mean, but ignore extremely large values
        nll = torch.clamp(nll, max=1e6)
        return nll.mean()  # Scalar loss

    def decision_scores_(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute anomaly scores per timestamp using NLL.
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
        
        # Apply same normalization as during training
        mean = np.mean(X_squeezed, axis=0, keepdims=True)
        std = np.std(X_squeezed, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
        X_normalized = (X_squeezed - mean) / std
        
        # Handle NaN or infinite values
        if np.isnan(X_normalized).any() or np.isinf(X_normalized).any():
            X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_windows = self._sliding_window(X_normalized)  # (N, W, C)
        X_tensor = torch.from_numpy(X_windows).float().to(self.device)

        self.model.eval()
        try:
            with torch.no_grad():
                mu, sigma = self.model(X_tensor)  # (N, W, C)
                # Ensure numerical stability
                sigma = torch.clamp(sigma + 1e-6, min=1e-6, max=1e6)
                mu = torch.clamp(mu, -1e6, 1e6)
                
                # Check for invalid values
                if torch.isnan(mu).any() or torch.isnan(sigma).any():
                    warnings.warn("Model produced NaN outputs during inference. Returning zeros...")
                    return scores_per_timestamp
                
                gaussian = dist.Normal(mu, sigma)
                nll = -gaussian.log_prob(X_tensor)  # (N, W, C)
                # Clamp extreme values
                nll = torch.clamp(nll, max=1e6)
                window_scores = nll.mean(dim=[1, 2])  # (N,) — avg over time & features

            window_scores = window_scores.cpu().numpy()

            # Assign score to last timestamp of each window
            for i, score in enumerate(window_scores):
                t_end = i * self.stride + self.window_size - 1
                if t_end < T:
                    scores_per_timestamp[t_end] = score

            # Forward-fill for robustness (optional but recommended)
            for i in range(1, T):
                if scores_per_timestamp[i] == 0 and scores_per_timestamp[i - 1] != 0:
                    scores_per_timestamp[i] = scores_per_timestamp[i - 1]

        except Exception as e:
            warnings.warn(f"Error during inference: {str(e)}. Returning zeros...")
            
        return scores_per_timestamp


class SensitiveHUE_NN(nn.Module):
    """
    Probabilistic GRU-based encoder that outputs μ and σ for each timestep.
    Input:  (B, W, C)
    Output: μ (B, W, C), σ (B, W, C)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc_mu = nn.Linear(hidden_dim, input_dim)
        self.fc_sigma = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights to prevent initial instability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, W, C)
        # Clamp input to prevent initial instability
        x = torch.clamp(x, -1e6, 1e6)
        gru_out, _ = self.gru(x)  # (B, W, H)
        mu = self.fc_mu(gru_out)   # (B, W, C)
        log_sigma = self.fc_sigma(gru_out)  # (B, W, C)
        sigma = torch.exp(torch.clamp(log_sigma, max=10))        # Ensure positivity and prevent overflow
        return mu, sigma