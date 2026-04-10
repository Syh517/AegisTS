import numpy as np

def print_npz_contents(file_path):
    """Load and print the contents of an NPZ file."""
    # Load the .npz file with allow_pickle=True to handle object arrays
    data = np.load(file_path, allow_pickle=True)
    
    print(f"Contents of {file_path}:")
    print("="*50)
    
    # Print the keys (names of arrays stored in the file)
    print("Keys in the file:", list(data.keys()))
    print()
    
    # Print information about each array in the file
    for key in data.keys():
        array = data[key]
        print(f"Key: '{key}'")
        print(f"Shape: {array.shape}")
        print(f"Data type: {array.dtype}")
        print(f"Number of dimensions: {array.ndim}")
        
        # Print a sample of the data (first few elements/values)
        print("Sample data:")
        if array.size > 0:
            if array.ndim == 1:
                # For 1D arrays, print first 10 elements
                print(array[:min(10, len(array))])
            elif array.ndim == 2:
                # For 2D arrays, print first few rows and columns
                rows = min(10, array.shape[0])
                cols = min(20, array.shape[1])
                print(array[:rows, :cols])
            else:
                # For higher dimensional arrays, print a smaller slice
                print("First few values (multi-dimensional):")
                print(array.flat[:min(50, array.size)])
        print("-" * 30)
    
    # Close the file handle
    data.close()

if __name__ == "__main__":
    file_path = "Datasets/Classification/merged/UWaveGestureLibrary.npz"
    print_npz_contents(file_path)