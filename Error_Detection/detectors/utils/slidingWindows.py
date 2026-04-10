from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf


from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np

def find_length_rank(data, rank=1, method='strongest'):
    """
    估计时间序列的主导周期长度（支持单变量和多变量）
    
    Parameters:
    -----------
    data : array-like, shape (N,) or (N, D)
        时间序列数据
    rank : int
        返回第几强的周期性模式（1=最强，2=次强...）
    method : str, {'mode', 'median', 'strongest', 'max'}
        多变量时如何聚合结果：
        - 'mode': 返回最频繁出现的周期（推荐）
        - 'median': 返回中位数
        - 'strongest': 返回 ACF 最强的那个周期
        - 'max': 返回最大周期
    
    Returns:
    --------
    int : 估计的周期长度，若无法确定则返回 20（默认值）
    """
    data = np.asarray(data).squeeze()

    # 处理单变量情况
    if data.ndim == 1:
        return find_length_rank_single(data, rank)

    # 处理多变量情况 (N, D)
    elif data.ndim == 2:
        periods = []
        strengths = []  # 记录每个变量最强周期处的 ACF 值（用于 strongest 策略）

        for col in range(data.shape[1]):
            ts = data[:, col]
            period = find_length_rank_single(ts, rank)
            if 3 <= period <= 300:  # 只保留合理范围内的周期
                periods.append(period)
                # 可选：计算该周期处的 ACF 强度
                acf_vals = acf(ts, nlags=400, fft=True)
                strength = acf_vals[period] if period < len(acf_vals) else 0
                strengths.append(strength)
            # else:
            #     strengths.append(0)

        if len(periods) == 0:
            return 20

        # 聚合策略
        if method == 'mode':
            # 使用众数，若无重复则用中位数
            vals, counts = np.unique(periods, return_counts=True)
            if np.max(counts) > 1:
                return int(vals[np.argmax(counts)])
            else:
                return int(np.median(periods))

        elif method == 'median':
            return int(np.median(periods))

        elif method == 'strongest':
            # 修复：添加检查确保strengths列表不为空
            if len(strengths) == 0:
                return 20
            return periods[np.argmax(strengths)]

        elif method == 'max':
            return int(np.max(periods))

        else:
            raise ValueError("method must be 'mode', 'median', 'strongest', or 'max'")

    else:
        return 20  # 不支持更高维



# determine sliding window (period) based on ACF
def find_length_rank_single(data, rank=1):
    data = data.squeeze()
    if len(data.shape) > 1: 
        return 0
    if rank == 0: 
        return 1
    data = data[:min(20000, len(data))]
    
    # 检查数据是否为空、全为常数或几乎无变化，避免除零错误
    if len(data) == 0 or np.std(data) == 0 or np.all(data == data[0]):
        return 20
        
    # 检查是否有足够的数据点进行ACF计算
    if len(data) < 10:  # 至少需要10个数据点
        return min(20, max(3, len(data) // 2))
    
    base = 3
    # 使用errstate来忽略可能的数值警告，并在出现问题时返回默认值
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            # 动态设置nlags，不超过数据长度的一半
            nlags = min(400, len(data) // 2)
            if nlags < base:
                return 20
            auto_corr = acf(data, nlags=nlags, fft=True)[base:]
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return 20
    
    # 检查auto_corr是否包含有效的数值
    if len(auto_corr) == 0 or np.all(np.isnan(auto_corr)) or np.any(np.isinf(auto_corr)):
        return 20
    
    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        # print('local_max[max_local_max]: ', local_max[max_local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 20
        return local_max[max_local_max]+base
    except:
        return 20
    

# determine sliding window (period) based on ACF, Original version
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 20
        return local_max[max_local_max]+base
    except:
        return 20
