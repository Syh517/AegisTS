import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, pred_len=10, n_layers=2,
                 lr=1e-3, n_epochs=100, batch_size=32, dropout=0.2):
        super(LSTMForecast, self).__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout

        # 自动选择 hidden_dim，如果没给定，按经验设置
        if hidden_dim is None:
            self.hidden_dim = max(32, min(128, input_dim * 8))
        else:
            self.hidden_dim = hidden_dim

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(self.hidden_dim, input_dim * pred_len)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # 标准化器
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # 通过全连接层得到预测结果
        output = self.fc(lstm_out)
        # 重塑为 (batch_size, pred_len, input_dim)
        output = output.view(-1, self.pred_len, self.input_dim)
        return output

    def fit(self, X_train, y_train):
        """训练模型"""
        # 数据验证
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
            
        # 数据标准化 - 分别处理输入和输出
        n_samples, seq_len, n_features = X_train.shape
        
        # 标准化输入X
        X_reshaped = X_train.reshape(-1, self.input_dim)
        X_scaled = self.scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        
        # 标准化目标y
        y_reshaped = y_train.reshape(-1, self.input_dim)
        y_scaled = self.scaler_y.fit_transform(y_reshaped).reshape(len(y_train), self.pred_len, self.input_dim)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # 训练模型
        self.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(X_tensor.size(0))
            
            for i in range(0, X_tensor.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_X, batch_y = X_tensor[indices], y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'LSTMForecast Epoch [{epoch+1}/{self.n_epochs}], Loss: {epoch_loss:.4f}')
                
        return self

    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            # 标准化输入
            original_shape_X = X.shape
            X_flat = X.reshape(-1, self.input_dim)
            X_scaled = self.scaler_X.transform(X_flat).reshape(original_shape_X)
            
            # 转换为PyTorch张量
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # 批量预测
            predictions = []
            for i in range(0, X_tensor.size(0), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_pred = self.forward(batch_X)
                predictions.append(batch_pred.cpu().numpy())
                
            # 合并所有批次的预测结果
            y_pred_scaled = np.concatenate(predictions, axis=0)
            
            # 反标准化预测结果
            original_shape_pred = y_pred_scaled.shape
            y_pred_flat = y_pred_scaled.reshape(-1, self.input_dim)
            y_pred_original = self.scaler_y.inverse_transform(y_pred_flat).reshape(original_shape_pred)
            
            return y_pred_original

    def save(self, filepath):
        """保存模型到文件"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'pred_len': self.pred_len,
            'n_layers': self.n_layers,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, filepath + '.pth')

    @classmethod
    def load(cls, filepath):
        """从文件加载模型"""
        checkpoint = torch.load(filepath + '.pth')
        
        # 创建模型实例
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            pred_len=checkpoint['pred_len'],
            n_layers=checkpoint['n_layers'],
            batch_size=checkpoint['batch_size'],
            dropout=checkpoint['dropout']
        )
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载标准化器
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
        
        return model