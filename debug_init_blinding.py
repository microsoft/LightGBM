import numpy as np
import tempfile
import subprocess
import os

def test_debug():
    # Generate constant features data
    n_samples, n_features = 20, 3
    X = np.ones((n_samples, n_features))
    y = np.ones(n_samples)
    
    # Save to temp file
    fd, data_file = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    
    try:
        with open(data_file, 'w') as f:
            for i in range(len(y)):
                line = f"{y[i]}"
                for j in range(X.shape[1]):
                    if X[i, j] != 0:
                        line += f" {j+1}:{X[i, j]}"
                f.write(line + "\n")
        
        print("Testing with minimal blinding (0.01)...")
        # Run LightGBM WITH minimal blind_volume
        cmd = [
            './lightgbm', 'task=train', f'data={data_file}',
            'output_model=debug_model.txt', 'num_iterations=1',  # Just 1 iteration
            'min_data_in_leaf=1', 'min_data_in_bin=1', 'blind_volume=0.01',  # Very small
            'verbosity=2'  # More verbose
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        return result.returncode
        
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)
        if os.path.exists('debug_model.txt'):
            os.unlink('debug_model.txt')

if __name__ == "__main__":
    test_debug()
