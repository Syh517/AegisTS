import numpy as np
import pandas as pd

from Datasets.load_dataset import load_single_dataset
from Error_Injection.injector import DataManager


def clean_timestamp_column_with_regression(data: np.ndarray) -> np.ndarray:
    """
    Clean timestamp column (first column) ensuring strict equidistant time intervals.
    
    Since we know the timestamp should be an equidistant increasing function without duplicates
    or fractional values, this function repairs any missing or corrupted values by reconstructing
    a perfectly equidistant sequence based on valid values.
    
    Args:
        data: 3D array of shape (N, T, D) where N is samples, T is timesteps, D is features
        
    Returns:
        Data with timestamp column cleaned and converted to integer values
    """
    data_cleaned = data.copy()
    n_samples, n_timesteps, n_features = data_cleaned.shape
    
    # Process timestamp column (index 0) for each sample
    for i in range(n_samples):
        timestamp_col = data_cleaned[i, :, 0].copy()
        
        # Find all valid (non-missing) values
        valid_mask = ~np.isnan(timestamp_col)
        valid_indices = np.where(valid_mask)[0]
        valid_values = timestamp_col[valid_indices]
        
        # If we have no valid values, we cannot repair
        if len(valid_values) == 0:
            continue
            
        # If we have only one value, fill everything with constant value
        if len(valid_values) == 1:
            timestamp_col[:] = valid_values[0]
            # Convert to integer
            data_cleaned[i, :, 0] = np.round(timestamp_col).astype(int)
            continue
            
        # For strict equidistant sequences, we calculate the step based on 
        # the difference between consecutive valid values if possible
        step = None
        if len(valid_values) >= 2:
            # Check for consecutive valid values to determine step size
            for j in range(len(valid_indices) - 1):
                # If indices are consecutive, calculate step
                if valid_indices[j+1] == valid_indices[j] + 1:
                    step = valid_values[j+1] - valid_values[j]
                    break
                    
        # If we couldn't find consecutive values to determine step,
        # estimate step from the overall sequence
        if step is None:
            if len(valid_indices) > 1:
                total_index_diff = valid_indices[-1] - valid_indices[0]
                total_value_diff = valid_values[-1] - valid_values[0]
                if total_index_diff > 0:
                    step = total_value_diff / total_index_diff
                else:
                    step = 0
            else:
                step = 0
        
        # Calculate start value and index
        start_index = valid_indices[0]
        start_value = valid_values[0]
        
        # Generate perfectly equidistant timestamp sequence
        # Using the formula: timestamp[i] = start_value + (i - start_index) * step
        ideal_timestamps = start_value + (np.arange(n_timesteps) - start_index) * step
        
        # Replace all values with ideal equidistant values
        timestamp_col[:] = ideal_timestamps
        
        # Convert to integer values to ensure no fractional parts
        data_cleaned[i, :, 0] = np.round(timestamp_col).astype(int)
        
    return data_cleaned




if __name__ == "__main__":
    type = 'forecast'
    dataset_name = 'weather'

    data, label = load_single_dataset(type, dataset_name, rate=0.2)

    if type == 'forecast':
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        print(data.shape)
        dm = DataManager(data, abnormal_rate=0.1, task_type=type)
    else:
        print(data.shape, label.shape)
        dm = DataManager(data, label, abnormal_rate=0.1, task_type=type)

    dm.inject_errors(0.1, ["missing", "duplicate"], covered_attrs=range(data.shape[-1]))

    data_dirty = dm.observed_data_restored.copy()
    
    # Clean timestamp using ideal linear assumption
    data_cleaned = clean_timestamp_column_with_regression(data_dirty)

    # Compare with ground truth
    clean_data = dm.clean_data_raw.copy()
    col1_clean = clean_data[0][:, 0]  # Ground truth
    col1_repaired = data_cleaned[0][:, 0]  # Repaired version

    # Find differences
    diff_mask = col1_clean != col1_repaired
    diff_indices = np.where(diff_mask)[0]

    print("不同的行索引:", diff_indices)
    print("对应值 (真实 vs 修复):")
    for i in diff_indices[:10]:
        print(f"  行 {i}: {col1_clean[i]} vs {col1_repaired[i]}")



        