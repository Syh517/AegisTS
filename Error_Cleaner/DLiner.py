import numpy as np
import torch
import torch.nn as nn

class DLinear:
    def __init__(self, seq_len, pred_len, num_features, lr=1e-3, n_epochs=200, batch_size=32):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 添加标准化参数
        self.mean = None
        self.std = None

    def _build_model(self):
        model = nn.Module()
        model.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        model.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        return model

    def fit(self, X_train, y_train):
        # 数据标准化
        if self.mean is None or self.std is None:
            self.mean = np.mean(X_train, axis=(0, 1), keepdims=True)
            self.std = np.std(X_train, axis=(0, 1), keepdims=True)
            self.std[self.std == 0] = 1  # 防止除以零
        
        X_train = (X_train - self.mean) / self.std
        y_train = (y_train - self.mean) / self.std
        
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        N = X_train.shape[0]
        
        for epoch in range(self.n_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            
            for i in range(0, N, self.batch_size):
                idx = perm[i:i+self.batch_size]
                x_batch = X_train[idx]
                y_batch = y_train[idx]
                
                self.optimizer.zero_grad()
                x = x_batch.permute(0,2,1)  # (batch, features, seq_len)
                s = self.model.linear_seasonal(x)
                t = self.model.linear_trend(x)
                y_pred = (s + t).permute(0,2,1)  # (batch, H, F)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item() * x_batch.size(0)
            
            avg_epoch_loss = epoch_loss / N
            if (epoch + 1) % 20 == 0:  # 每20个epoch打印一次
                print(f"DLinear Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_epoch_loss:.6f}")
                
        return self

    def predict(self, X):
        # 使用训练时的均值和标准差进行相同的标准化
        X = (X - self.mean) / self.std
        
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 使用批处理进行预测
        predictions = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                x_batch = X[i:i+self.batch_size]
                x = x_batch.permute(0,2,1)
                s = self.model.linear_seasonal(x)
                t = self.model.linear_trend(x)
                y_pred = (s + t).permute(0,2,1)
                predictions.append(y_pred.cpu().numpy())
                
        # 合并所有批次的结果，并进行反标准化
        result = np.concatenate(predictions, axis=0)
        result = result * self.std + self.mean
        return result