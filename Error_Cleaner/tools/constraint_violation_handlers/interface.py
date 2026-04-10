#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单接口函数，用于根据指定的清洗工具名称、约束和数据进行清洗
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List


# from Error_Cleaner.tools.constraint_violation_handlers.MTSClean_liner import MTSClean
from Error_Cleaner.tools.constraint_violation_handlers.MTSClean_poly import MTSClean
from Error_Cleaner.tools.constraint_violation_handlers.SpeedAcc import SpeedAcc
# from Error_Cleaner.tools.constraint_violation_handlers.Clean4MTS_liner import Clean4MTS
from Error_Cleaner.tools.constraint_violation_handlers.Clean4MTS_poly import Clean4MTS
from Error_Cleaner.tools.constraint_violation_handlers.MTCSC_N import MTCSC_N
from Error_Cleaner.tools.constraint_violation_handlers.HTD import HTDClean
from Error_Cleaner.tools.constraint_violation_handlers.Screen import SCREEN
from Error_Cleaner.tools.constraint_violation_handlers.OneMILP import OneMILP
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import speed_constraint_clean_local
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import speed_constraint_clean_global
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import speed_plus_acceleration_constraint_clean_local
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import speed_plus_acceleration_constraint_clean_global
from Error_Cleaner.tools.constraint_violation_handlers.benchmark import variance_constraint_clean

# 定义每种工具需要的约束类型映射
_TOOL_REQUIRED_CONSTRAINTS = {
    'mtclean': ['speed_constraints', 'accel_constraints', 'row_constraints'],
    'speedacc': ['speed_constraints', 'accel_constraints'],
    'clean4mts': ['speed_constraints', 'accel_constraints', 'row_constraints'],
    'mtcsc_n': ['speed_constraints'],
    'htd': ['speed_constraints'],
    'screen': ['speed_constraints'],
    'oneMilp': ['speed_constraints'],
    'speed_local': ['speed_constraints'],
    'speed_global': ['speed_constraints'],
    'speed_accel_local': ['speed_constraints', 'accel_constraints'],
    'speed_accel_global': ['speed_constraints', 'accel_constraints'],
    'variance': ['variance_constraints']
}

def clean_with_tool(tool_name: str, data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """
    使用指定的清洗工具清洗数据
    
    Args:
        tool_name: 清洗工具名称 (如 'mtclean', 'speedacc', 'clean4mts' 等)
        data: 需要清洗的数据
        constraints: 约束字典，格式为:
            {
                'row_constraints': [],
                'speed_constraints': {...},
                'accel_constraints': {...},
                'variance_constraints': {...},
                'od_constraints': [],
                'dd_constraints': []
            }
    
    Returns:
        清洗后的数据
    """
    
    # 处理约束中的numpy类型
    processed_constraints = _process_constraints(constraints)
    
    # 输出调试信息
    print(f"正在使用清洗工具: {tool_name}")
    
    # 检查是否支持该工具
    if tool_name not in _TOOL_REQUIRED_CONSTRAINTS:
        raise ValueError(f"不支持的清洗工具: {tool_name}")
    
    # 自动提取所需约束（仅传递该工具需要的部分）
    required_keys = _TOOL_REQUIRED_CONSTRAINTS[tool_name]
    filtered_constraints = {key: processed_constraints.get(key, {}) for key in required_keys}
    
    # 调试：显示实际使用的约束
    for key in required_keys:
        value = filtered_constraints[key]
        if value:
            print(f"  使用约束 [{key}]: {list(value.keys())}")
        else:
            print(f"  注意: 工具 '{tool_name}' 请求的约束 [{key}] 为空或未提供")
    
    # 根据工具名称调用对应的清洗函数（仍保持原有逻辑结构）
    if tool_name == 'mtclean':
        return _clean_with_mtclean(data, filtered_constraints)
    elif tool_name == 'speedacc':
        return _clean_with_speedacc(data, filtered_constraints)
    elif tool_name == 'clean4mts':
        return _clean_with_clean4mts(data, filtered_constraints)
    elif tool_name == 'mtcsc_n':
        return _clean_with_mtcsc_n(data, filtered_constraints)
    elif tool_name == 'htd':
        return _clean_with_htd(data, filtered_constraints)
    elif tool_name == 'screen':
        return _clean_with_screen(data, filtered_constraints)
    elif tool_name == 'oneMilp':
        return _clean_with_oneMilp(data, filtered_constraints)
    elif tool_name == 'speed_local':
        return _clean_with_speed_local(data, filtered_constraints)
    elif tool_name == 'speed_global':
        return _clean_with_speed_global(data, filtered_constraints)
    elif tool_name == 'speed_accel_local':
        return _clean_with_speed_accel_local(data, filtered_constraints)
    elif tool_name == 'speed_accel_global':
        return _clean_with_speed_accel_global(data, filtered_constraints)
    elif tool_name == 'variance':
        return _clean_with_variance(data, filtered_constraints)
    else:
        raise ValueError(f"不支持的清洗工具: {tool_name}")  # 冗余检查，理论上不会执行到这里


def _process_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理约束中的numpy类型数据
    
    Args:
        constraints: 原始约束字典
        
    Returns:
        处理后的约束字典
    """
    processed = {}
    
    for key, value in constraints.items():
        if key.endswith('_constraints') and isinstance(value, dict):
            # 处理约束字典中的numpy类型
            processed_constraints = {}
            for k, v in value.items():
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    # 转换numpy类型为普通float
                    lower = float(v[0]) if hasattr(v[0], 'item') else v[0]
                    upper = float(v[1]) if hasattr(v[1], 'item') else v[1]
                    processed_constraints[k] = (lower, upper)
                else:
                    processed_constraints[k] = v
            processed[key] = processed_constraints
        else:
            processed[key] = value
            
    return processed


def _clean_with_mtclean(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用MTSClean清洗数据"""
    try:
        cleaner = MTSClean()
        cleaned_data = cleaner.clean(
            data,
            speed_constraints=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            row_constraints=constraints.get('row_constraints', [])
        )
        return cleaned_data
    except Exception as e:
        print(f"使用MTSClean清洗数据时出错: {e}")
        return data


def _clean_with_speedacc(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用SpeedAcc清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_cols = [col for col in data.columns if col != time_col]
        
        # 创建清洗器实例
        cleaner = SpeedAcc(
            df=data,
            time_col=time_col,
            data_cols=data_cols,
            T=5.0,  # 默认时间窗口
            speed_constraints=constraints.get('speed_constraints', {}),
            acc_constraints=constraints.get('accel_constraints', {})
        )
        
        # 执行清洗
        result = cleaner.run()
        
        # 提取清洗后的数据列
        cleaned_data = data.copy()
        for col in data.columns:
            clean_col = f"{col}_clean"
            if clean_col in result.columns:
                cleaned_data[col] = result[clean_col].values
                
        return cleaned_data
    except Exception as e:
        print(f"使用SpeedAcc清洗数据时出错: {e}")
        return data

def _clean_with_clean4mts(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    try:
        # 安装提示：如果环境中没有 pygad，请执行 pip install pygad
        cleaner = Clean4MTS(
            dataframe=data,
            speed_constraint=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            row_constraints=constraints.get('row_constraints', [])
        )
        return cleaner.data_cleaning()
    except Exception as e:
        print(f"Clean4MTS (PyGAD) 运行出错: {e}")
        return data
    
# def _clean_with_clean4mts(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
#     """使用Clean4MTS清洗数据"""
#     try:
        
#         # 确定时间列
#         time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
#         # 获取数据列（除时间列外的其他列）
#         relevant_attrs = [col for col in data.columns if col != time_col]
        
#         # 创建清洗器实例
#         cleaner = Clean4MTS(
#             dataframe=data,
#             speed_constraint=constraints.get('speed_constraints', {}),
#             temporal_attr=time_col,
#             relevant_attrs=relevant_attrs
#         )
        
#         # 执行清洗
#         cleaned_data = cleaner.data_cleaning()
        
#         return cleaned_data
#     except Exception as e:
#         print(f"使用Clean4MTS清洗数据时出错: {e}")
#         return data

def _clean_with_mtcsc_n(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用MTCSC_N清洗数据"""
    try:
        cleaner = MTCSC_N(
            df=data,
            speed_constraints=constraints.get('speed_constraints', {}),
            t=5
        )
        result = cleaner.mainScreen()
        cleaned_data = data.copy()
        for col in data.columns:
            modify_col = f"{col}_modify"
            if modify_col in result.columns:
                cleaned_data[col] = result[modify_col].values
        return cleaned_data
    except Exception as e:
        print(f"使用MTCSC_N清洗数据时出错: {e}")
        return data
    

# def _clean_with_mtcsc_as(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
#     """使用MTCSC_AS清洗数据"""
#     try:
        
#         # 创建清洗器实例
#         cleaner = MTCSC_AS(
#             df=data,
#             speed_constraints=constraints.get('speed_constraints', {}),
#             t=5  # 默认时间窗口
#         )
        
#         # 执行清洗
#         result = cleaner.mainScreen()
        
#         # 提取清洗后的数据列
#         cleaned_data = data.copy()
#         for col in data.columns:
#             modify_col = f"{col}_modify"
#             if modify_col in result.columns:
#                 cleaned_data[col] = result[modify_col].values
                
#         return cleaned_data
#     except Exception as e:
#         print(f"使用MTCSC_AS清洗数据时出错: {e}")
#         return data


def _clean_with_htd(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用HTD清洗数据"""
    try:
        
        # 创建清洗器实例
        cleaner = HTDClean(
            speed_constraints=constraints.get('speed_constraints', {})
        )
        
        # 执行清洗
        cleaned_data = cleaner.clean(data)
        
        return cleaned_data
    except Exception as e:
        print(f"使用HTD清洗数据时出错: {e}")
        return data


def _clean_with_screen(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用SCREEN清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 创建清洗器实例
        cleaner = SCREEN(
            df=data,
            data_columns=data_columns,
            speed_constraints=constraints.get('speed_constraints', {}),
            window_size=5  # 默认时间窗口
        )
        
        # 执行清洗
        result = cleaner.main_screen()
        
        # 提取清洗后的数据列
        cleaned_data = data.copy()
        for col in data.columns:
            modify_col = f"{col}_modify"
            if modify_col in result.columns:
                cleaned_data[col] = result[modify_col].values
                
        return cleaned_data
    except Exception as e:
        print(f"使用SCREEN清洗数据时出错: {e}")
        return data


# def _clean_with_oneMilp(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
#     """使用OneMILP清洗数据"""
#     try:
        
#         # 创建清洗器实例
#         cleaner = OneMILP(
#             speed_constraints=constraints.get('speed_constraints', {}),
#             T=5
#         )
        
#         # 执行清洗
#         result = cleaner.mainGlobal(data)
        
#         # 提取清洗后的数据列
#         cleaned_data = data.copy()
#         for col in data.columns:
#             repaired_col = f"{col}_repaired"
#             if repaired_col in result.columns:
#                 cleaned_data[col] = result[repaired_col].values
                
#         return cleaned_data
#     except Exception as e:
#         print(f"使用OneMILP清洗数据时出错: {e}")
#         return data


def _clean_with_oneMilp(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """
    使用 NVarMILP 清洗数据
    
    :param data: 原始 DataFrame
    :param constraints: 包含 'speed_constraints' 的字典
    :return: 清洗后的 DataFrame
    """
    try:

        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        timestamps = data[time_col].values
        
        data_cols = [col for col in data.columns if col != time_col]
        df_obs = data[data_cols]
        
        
        cleaner = OneMILP(
            timestamps=timestamps,
            df_obs=df_obs,
            constraints=constraints.get('speed_constraints', {}),
            T_window=60
        )
        
        cleaned_values, _ = cleaner.clean_data_segmented(segment_size=80, overlap=10)
        
        if cleaned_values is not None:
            cleaned_data = data.copy()
            # 更新对应的数据列
            for idx, col in enumerate(data_cols):
                cleaned_data[col] = cleaned_values[:, idx]
            return cleaned_data
        else:
            return data

    except Exception as e:
        print(f"使用 OneMILP 清洗数据时出错: {e}")
        return data

def _clean_with_speed_local(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用本地速度约束清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 执行清洗
        cleaned_data = data.copy()
        speed_constraint_clean_local(
            dataframe=cleaned_data,
            speed_constraints=constraints.get('speed_constraints', {}),
            ra=data_columns,
            t_attr=time_col,
            w=50  # 默认窗口大小
        )
        
        return cleaned_data
    except Exception as e:
        print(f"使用本地速度约束清洗数据时出错: {e}")
        return data


def _clean_with_speed_global(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用全局速度约束清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 执行清洗
        cleaned_data = data.copy()
        speed_constraint_clean_global(
            dataframe=cleaned_data,
            speed_constraints=constraints.get('speed_constraints', {}),
            ra=data_columns,
            t_attr=time_col,
            w=50,  # 默认窗口大小
            size=100,  # 默认块大小
            overlapping_ratio=0.2  # 默认重叠率
        )
        
        return cleaned_data
    except Exception as e:
        print(f"使用全局速度约束清洗数据时出错: {e}")
        return data


def _clean_with_speed_accel_local(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用本地速度和加速度约束清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 执行清洗
        cleaned_data = data.copy()
        speed_plus_acceleration_constraint_clean_local(
            dataframe=cleaned_data,
            speed_constraints=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            ra=data_columns,
            t_attr=time_col,
            w=50  # 默认窗口大小
        )
        
        return cleaned_data
    except Exception as e:
        print(f"使用本地速度和加速度约束清洗数据时出错: {e}")
        return data


def _clean_with_speed_accel_global(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用全局速度和加速度约束清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 执行清洗
        cleaned_data = data.copy()
        speed_plus_acceleration_constraint_clean_global(
            dataframe=cleaned_data,
            speed_constraints=constraints.get('speed_constraints', {}),
            acceleration_constraints=constraints.get('accel_constraints', {}),
            ra=data_columns,
            t_attr=time_col,
            w=50,  # 默认窗口大小
            size=100,  # 默认块大小
            overlapping_ratio=0.1  # 默认重叠率
        )
        
        return cleaned_data
    except Exception as e:
        print(f"使用全局速度和加速度约束清洗数据时出错: {e}")
        return data


def _clean_with_variance(data: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """使用方差约束清洗数据"""
    try:
        
        # 确定时间列
        time_col = 'timestamp' if 'timestamp' in data.columns else data.columns[0]
        
        # 获取数据列（除时间列外的其他列）
        data_columns = [col for col in data.columns if col != time_col]
        
        # 执行清洗
        cleaned_data = data.copy()
        variance_constraint_clean(
            dataframe=cleaned_data,
            variance_constraints=constraints.get('variance_constraints', {}),
            ra=data_columns,
            t_attr=time_col,
            w=10,  # 默认窗口大小
            beta=0.5  # 默认衰减因子
        )
        
        return cleaned_data
    except Exception as e:
        print(f"使用方差约束清洗数据时出错: {e}")
        return data


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    import numpy as np
    np.random.seed(42)
    n_points = 100
    
    data = pd.DataFrame({
        'timestamp': np.arange(n_points),
        'col_1': np.random.randn(n_points).cumsum(),
        'col_2': np.random.randn(n_points).cumsum(),
        'col_3': np.random.randn(n_points).cumsum()
    })
    
    # 人为引入一些违反约束的点
    data.loc[20, 'col_1'] = data.loc[19, 'col_1'] + 15.0
    data.loc[45, 'col_2'] = data.loc[44, 'col_2'] - 8.0
    
    # 定义约束
    constraints = {
        'row_constraints': [],
        'speed_constraints': {
            'col_1': (np.float64(-2.0), np.float64(2.0)),
            'col_2': (np.float64(-1.5), np.float64(1.5)),
            'col_3': (np.float64(-3.0), np.float64(3.0))
        },
        'accel_constraints': {
            'col_1': (np.float64(-0.5), np.float64(0.5)),
            'col_2': (np.float64(-0.3), np.float64(0.3)),
            'col_3': (np.float64(-0.7), np.float64(0.7))
        },
        'variance_constraints': {
            'col_1': (np.float64(0.0), np.float64(10.0)),
            'col_2': (np.float64(0.0), np.float64(5.0)),
            'col_3': (np.float64(0.0), np.float64(8.0))
        },
        'od_constraints': [],
        'dd_constraints': []
    }
    
    # 使用指定的工具清洗数据
    print("使用MTSClean清洗数据...")
    cleaned_data = clean_with_tool('clean4mts', data.copy(), constraints)
    print("清洗完成!")
    print(f"原始数据形状: {data.shape}")
    print(f"清洗后数据形状: {cleaned_data.shape}")

    print("\n原始数据在 index 30:")
    print(data.iloc[30])
    print("\n仅速度修复后:")
    print(cleaned_data.iloc[30])
    print("\n速度+加速度修复后:")
    print(cleaned_data.iloc[30])

    