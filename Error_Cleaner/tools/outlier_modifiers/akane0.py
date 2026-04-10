import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline 


# ---------------------- 核心组件：PatternMiner ----------------------
class MVPatternMiner:
    """多变量模式发现：基于滑窗+聚类的子序列模式发现。"""
    def __init__(self, window_len=20, n_patterns=5, stride=1):
        self.window_len = window_len
        self.n_patterns = n_patterns
        self.stride = stride
        self.patterns = None
        self.labels = None
        self.km = None
        self.idxs = None 
        self.n_features = None

    def _extract_windows(self, X):
        # X: (n_steps, n_features) numpy array
        n, d = X.shape
        self.n_features = d
        w = self.window_len
        idxs = []
        windows = []
        
        # 窗口提取：将 (window_len, n_features) 窗口展平为 (window_len * n_features) 向量
        for i in range(0, n - w + 1, self.stride):
            window = X[i:i+w, :]
            windows.append(window.flatten()) 
            idxs.append(i)
            
        if not windows:
            return np.array([]).reshape(0, w * d), np.array([])
        return np.vstack(windows), np.array(idxs)

    def _internal_fit(self, windows_z, k):
        km = KMeans(n_clusters=k, random_state=0, n_init='auto')
        km.fit(windows_z)
        return km

    def auto_fit(self, X, k_range=(5, 15)):
        """使用轮廓系数法自动选择最佳 N_PATTERNS (K)"""
        X = np.asarray(X).astype(float)
        windows, idxs = self._extract_windows(X)
        self.idxs = idxs
        
        min_k, max_k = k_range
        if len(windows) < max_k: max_k = len(windows) - 1
        if min_k >= max_k or max_k < 3:
            self.n_patterns = max(1, min_k)
            self.fit(X)
            return

        # Z-score normalization per flattened window
        w_mean = windows.mean(axis=1, keepdims=True)
        w_std = windows.std(axis=1, keepdims=True) + 1e-8
        windows_z = (windows - w_mean) / w_std

        best_k = min_k
        best_score = -1
        
        for k in range(min_k, max_k + 1):
            try:
                km = self._internal_fit(windows_z, k)
                if len(np.unique(km.labels_)) < 2: continue
                score = silhouette_score(windows_z, km.labels_)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        
        self.n_patterns = best_k
        print(f"MVPatternMiner: 自动选择 N_PATTERNS = {best_k} (Silhouette Score: {best_score:.4f})")
        
        self.fit(X) 
        
    def fit(self, X):
        X = np.asarray(X).astype(float)
        windows, idxs = self._extract_windows(X)
        self.idxs = idxs
        if len(windows) == 0:
            self.labels = np.zeros(len(X), dtype=int) - 1
            return

        w_mean = windows.mean(axis=1, keepdims=True)
        w_std = windows.std(axis=1, keepdims=True) + 1e-8
        windows_z = (windows - w_mean) / w_std

        k = min(self.n_patterns, len(windows_z))
        if k <= 1:
            self.patterns = [windows.mean(axis=0)]
            self.labels = np.zeros(len(X), dtype=int) - 1
            return

        km = self._internal_fit(windows_z, k)
        cluster_ids = km.labels_
        self.km = km
        self.patterns = km.cluster_centers_

        label_series = -1 * np.ones(len(X), dtype=int)
        for wi, start in enumerate(idxs):
            cid = cluster_ids[wi]
            label_series[start:start+self.window_len] = cid 
        self.labels = label_series

    def get_labels(self):
        return self.labels

# ---------------------- 核心组件：Repairer (单变量辅助，多变量遍历) ----------------------
class MVRepairer:
    """多变量修复策略：遍历所有特征，对每个特征应用单变量修复方法。"""
    
    @staticmethod
    def _univariate_local_linear(x, idx, window):
        n = len(x)
        L = window
        lo = max(0, idx-L)
        hi = min(n, idx+L+1)
        xs = np.arange(lo, hi).reshape(-1,1)
        ys = x[lo:hi]
        
        mask = np.ones(len(xs), dtype=bool)
        mask[xs.flatten() == idx] = False
        mask[np.isnan(ys)] = False

        if mask.sum() < 2:
            # 简单的邻居均值作为回退
            return (x[idx-1] + x[idx+1]) / 2 if 0 < idx < n-1 else x[idx]
        
        model = LinearRegression()
        model.fit(xs[mask], ys[mask])
        return model.predict(np.array([[idx]])).item()

    @staticmethod
    def _univariate_cubic_spline(x, idx, window):
        n = len(x)
        L = window
        lo = max(0, idx-L)
        hi = min(n, idx+L+1)
        xs_full = np.arange(lo, hi)
        ys_full = x[lo:hi]
        
        mask = np.ones(len(xs_full), dtype=bool)
        mask[xs_full == idx] = False
        mask[np.isnan(ys_full)] = False
        
        xs_fit = xs_full[mask]
        ys_fit = ys_full[mask]
        
        if len(xs_fit) < 4:
            return MVRepairer._univariate_local_linear(x, idx, window)

        try:
            cs = CubicSpline(xs_fit, ys_fit, bc_type='natural') 
            return cs(idx).item()
        except Exception:
            return MVRepairer._univariate_local_linear(x, idx, window)

    @staticmethod
    def local_linear(X, idx, window=5):
        """修复所有特征"""
        d = X.shape[1]
        repaired_values = np.zeros(d)
        for feature_idx in range(d):
            repaired_values[feature_idx] = MVRepairer._univariate_local_linear(X[:, feature_idx], idx, window)
        return repaired_values

    @staticmethod
    def cubic_spline(X, idx, window=10):
        """修复所有特征"""
        d = X.shape[1]
        repaired_values = np.zeros(d)
        for feature_idx in range(d):
            repaired_values[feature_idx] = MVRepairer._univariate_cubic_spline(X[:, feature_idx], idx, window)
        return repaired_values

# ---------------------- 核心组件：PerplexityModel (与单变量一致，基于 Pattern ID 序列) ----------------------
class PerplexityModel:
    """提供 N-阶马尔可夫链（基于 pattern id 序列）"""
    # 保持与前面提供的 N-阶 Markov 模型代码一致
    def __init__(self, backend='markov', n_components=5, markov_order=1):
        assert backend == 'markov' # 多变量Akane只使用markov
        self.backend = backend
        self.n_components = n_components
        self.markov_order = markov_order
        self.trans_mat = None
        self.n_states = n_components

    def fit_markov(self, states, n_states=None):
        states = np.asarray(states).astype(int)
        valid_states = states[states >= 0]
        if n_states is None: n_states = int(valid_states.max() + 1) if len(valid_states) > 0 else self.n_states
        self.n_states = n_states
        trans_freq = {} 
        if len(valid_states) < self.markov_order + 1:
            self.trans_mat = {}
            return

        for i in range(len(valid_states) - self.markov_order):
            history = tuple(valid_states[i:i + self.markov_order])
            current = valid_states[i + self.markov_order]
            if history not in trans_freq: trans_freq[history] = {}
            trans_freq[history][current] = trans_freq[history].get(current, 0) + 1

        self.trans_mat = {}
        smoothing = 1e-6
        for history, next_counts in trans_freq.items():
            total = sum(next_counts.values())
            smoothed_total = total + self.n_states * smoothing
            probs = {s: smoothing / smoothed_total for s in range(self.n_states)}
            for next_state, count in next_counts.items():
                probs[next_state] = (count + smoothing) / smoothed_total
            self.trans_mat[history] = probs

    def score_sequence_markov(self, states):
        states = np.asarray(states).astype(int)
        valid_states = states[states >= 0]
        logp = 0.0
        count = 0
        if not self.trans_mat or len(valid_states) < self.markov_order + 1: return -1e9

        for i in range(len(valid_states) - self.markov_order):
            history = tuple(valid_states[i:i + self.markov_order])
            current = valid_states[i + self.markov_order]
            if history in self.trans_mat and current in self.trans_mat[history]:
                p = self.trans_mat[history][current]
                logp += np.log(max(p, 1e-12))
                count += 1
            else:
                logp += np.log(1e-12) 
                count += 1

        if count == 0: return -1e9 
        return logp / count

    def perplexity(self, x, states=None):
        avg_logp = self.score_sequence_markov(states)
        ppl = float(np.exp(-avg_logp))
        return ppl

# ---------------------- 核心组件：Cleaner ----------------------
class MultivariateCleaner:
    """限制只对标签为1的点进行清洗的多变量 Akane 贪婪选择器。"""
    def __init__(self, backend='markov', n_components=5, markov_order=1, 
                 repairer=None, cost_fn=None, miner=None,
                 candidate_indices=None): # <--- 关键参数
        
        # 确保修复器和模式发现器已提供
        assert repairer is not None and miner is not None, "Repairer and Miner must be provided."

        self.perp = PerplexityModel(backend=backend, n_components=n_components, markov_order=markov_order)
        # 注意：cost_fn 在这个强制修复模式下实际上没有约束作用，但保留其定义
        self.repairer = repairer
        self.cost_fn = cost_fn if cost_fn is not None else (lambda idx: 1.0)
        self.miner = miner
        self.candidate_indices = candidate_indices # 限制清洗的索引（NumPy 索引）
        self.initial_ppl = None

    def _get_current_states(self, X_current, miner_instance):
        """重新计算 Pattern ID 序列"""
        windows, idxs = miner_instance._extract_windows(X_current)
        if len(windows) == 0 or miner_instance.km is None:
             return -1 * np.ones(X_current.shape[0], dtype=int)
            
        w_mean = windows.mean(axis=1, keepdims=True)
        w_std = windows.std(axis=1, keepdims=True) + 1e-8
        windows_z = (windows - w_mean) / w_std

        # 映射到现有模式
        distances = cdist(windows_z, miner_instance.km.cluster_centers_)
        cluster_ids = np.argmin(distances, axis=1)

        label_series = -1 * np.ones(X_current.shape[0], dtype=int)
        for wi, start in enumerate(idxs):
            cid = cluster_ids[wi]
            label_series[start:start+miner_instance.window_len] = cid
        return label_series
    
    def greedy_clean(self, X, budget=None): # budget 参数在此逻辑中被忽略
        """
        清洗逻辑：强制修复所有 label=1 的候选点，并根据困惑度增益排序决定修复顺序。
        """
        X = np.asarray(X).astype(float)
        X_current = X.copy()
        
        # 初始拟合和困惑度计算
        initial_labels = self.miner.get_labels()
        self.perp.fit_markov(initial_labels)
        base_ppl = self.perp.perplexity(X_current, initial_labels)
        self.initial_ppl = base_ppl
        
        selected = []
        
        # 候选池：仅包含 label=1 且非 NaN 的索引
        candidate_pool = [i for i in self.candidate_indices if not np.any(np.isnan(X_current[i, :]))]

        # 如果没有有效的候选点，直接返回
        if not candidate_pool:
            return X_current, selected, base_ppl

        print(f"Cleaner: 发现 {len(candidate_pool)} 个有效的强制修复候选点。")

        # ------------------- 阶段 1: 计算所有候选点的初始修复收益 -------------------
        
        gain_scores = []
        # 预计算所有修复操作的收益
        for idx in candidate_pool:
            # 模拟修复
            X_sim = X_current.copy()
            repaired_val = self.repairer(X_sim, idx) 
            X_sim[idx, :] = repaired_val 
            
            # 困惑度计算是整个算法中计算量最大的部分
            sim_states = self._get_current_states(X_sim, self.miner)
            ppl_sim = self.perp.perplexity(X_sim, sim_states)
            
            # 增益 = 困惑度降低
            gain = base_ppl - ppl_sim
            gain_scores.append({'idx': idx, 'gain': gain, 'repaired_val': repaired_val})

        # ------------------- 阶段 2: 排序并依次应用修复 -------------------
        
        # 按收益降序排序（优先修复最能降低困惑度的点）
        gain_scores.sort(key=lambda x: x['gain'], reverse=True)
        
        # 依次应用修复，并更新模型状态
        for item in gain_scores:
            idx = item['idx']
            repaired_val = item['repaired_val']
            
            # 应用修复值
            X_current[idx, :] = repaired_val
            
            # 记录修复信息
            selected.append((idx, repaired_val, item['gain'])) # 记录的是基于上次模型的增益
            
            # 重新拟合 Markov 模型并更新基准困惑度 (这是贪婪选择的关键)
            new_labels = self._get_current_states(X_current, self.miner)
            self.perp.fit_markov(new_labels) 
            base_ppl = self.perp.perplexity(X_current, new_labels)

        # 修复总数为 label=1 的点数
        final_ppl = base_ppl
        return X_current, selected, final_ppl

