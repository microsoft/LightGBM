import numpy as np
import tempfile
import subprocess
import os

def test_constant_features_original():
    """Test constant features on original LightGBM (without blind_volume)"""
    print("Testing constant features on original LightGBM...")
    
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
        
        # Run LightGBM WITHOUT blind_volume
        cmd = [
            './lightgbm', 'task=train', f'data={data_file}',
            'output_model=debug_model.txt', 'num_iterations=5',
            'min_data_in_leaf=1', 'min_data_in_bin=1'
            # Note: NO blind_volume parameter
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")
        else:
            print("✅ Original LightGBM handles constant features successfully")
            
        return result.returncode
        
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)
        if os.path.exists('debug_model.txt'):
            os.unlink('debug_model.txt')

def test_constant_features_with_blinding():
    """Test constant features WITH blinding"""
    print("\nTesting constant features WITH blinding...")
    
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
        
        # Run LightGBM WITH blind_volume
        cmd = [
            './lightgbm', 'task=train', f'data={data_file}',
            'output_model=debug_model.txt', 'num_iterations=5',
            'min_data_in_leaf=1', 'min_data_in_bin=1', 'blind_volume=0.4'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")
            if result.returncode == -11:
                print("❌ Segmentation fault - our blinding implementation has an issue")
            else:
                print("❌ Other error")
        else:
            print("✅ Blinding handles constant features successfully")
            
        return result.returncode
        
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)
        if os.path.exists('debug_model.txt'):
            os.unlink('debug_model.txt')

if __name__ == "__main__":
    original_result = test_constant_features_original()
    blinding_result = test_constant_features_with_blinding()
    
    print(f"\n--- Summary ---")
    print(f"Original LightGBM: {'✅ PASS' if original_result == 0 else '❌ FAIL'}")
    print(f"With blinding: {'✅ PASS' if blinding_result == 0 else '❌ FAIL'}")
    
    if original_result == 0 and blinding_result != 0:
        print("🔍 Our blinding implementation introduced a regression!")
    elif original_result != 0 and blinding_result != 0:
        print("ℹ️  Both fail - this is a pre-existing edge case")
    elif blinding_result == 0:
        print("✅ All working correctly!")
