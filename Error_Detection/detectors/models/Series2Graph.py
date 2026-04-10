import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAutoEncoder(nn.Module):
    def __init__(self, n_nodes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(n_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_nodes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        x_hat = self.fc2(h)
        return x_hat

class Series2Graph:
    """
    多变量时序异常检测（Series2Graph + Graph AutoEncoder）
    - data: numpy.ndarray, shape (n_samples, n_features)
    - window: 滑动窗口大小
    """
    def __init__(self, window=10, hidden_dim=32, lr=1e-3, epochs=50):
        self.window = window
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.decision_scores_ = None

    def _build_graph(self, window_data):
        """
        构建图的邻接矩阵
        - window_data: (window, n_features)
        返回节点特征矩阵：每个节点的平均值
        """
        node_feat = np.mean(window_data, axis=0)  # n_features
        return node_feat

    def fit(self, data: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_samples, n_features = data.shape
        scores = np.zeros(n_samples)

        # 构建所有窗口节点特征
        node_features = []
        for t in range(self.window, n_samples):
            window_data = data[t-self.window:t]
            node_feat = self._build_graph(window_data)
            node_features.append(node_feat)
        node_features = np.stack(node_features)  # shape (n_samples-window, n_features)

        # 训练 Graph AutoEncoder
        model = GraphAutoEncoder(n_nodes=n_features, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        X_tensor = torch.tensor(node_features, dtype=torch.float32).to(device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            X_hat = model(X_tensor)
            loss = F.mse_loss(X_hat, X_tensor)
            loss.backward()
            optimizer.step()

        # 计算异常分数
        with torch.no_grad():
            X_hat = model(X_tensor).cpu().numpy()
        residual = np.abs(X_hat - node_features)
        max_residual = np.max(residual, axis=1)
        # 填充前 window 个点为0
        scores[:self.window] = 0
        scores[self.window:] = max_residual
        self.decision_scores_ = scores
        return self
