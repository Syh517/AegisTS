import pandas as pd
import numpy as np

def rank_model_performance(csv_file_path):
    """
    读取性能矩阵CSV文件，计算所有模型的平均性能并进行排名
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 获取模型列（除了第一列'Dataset'之外的所有列）
    model_columns = df.columns[1:]
    
    # 计算每个模型的平均性能
    model_averages = {}
    for model in model_columns:
        model_averages[model] = df[model].mean()
    
    # 将结果转换为DataFrame以便排序
    averages_df = pd.DataFrame.from_dict(model_averages, orient='index', columns=['Average Performance'])
    
    # 按平均性能降序排列
    ranked_df = averages_df.sort_values(by='Average Performance', ascending=False)
    ranked_df['Rank'] = range(1, len(ranked_df) + 1)
    
    return ranked_df

if __name__ == "__main__":
    csv_file_path = "/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results_val/performance_matrix_vus_pr.csv"
    
    # 计算并显示排名结果
    result = rank_model_performance(csv_file_path)
    
    print("模型性能排名:")
    print(result)
    
    print("\n前10名模型:")
    print(result.head(10))
    
    print("\n后10名模型:")
    print(result.tail(10))