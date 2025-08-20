import numpy as np
import subprocess
import tempfile
import os

def create_test_data():
    """Create simple test dataset"""
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some relationship to features
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    return X, y

def save_libsvm_format(X, y, filename):
    """Save data in LibSVM format"""
    with open(filename, 'w') as f:
        for i in range(len(y)):
            line = f"{y[i]}"
            for j in range(X.shape[1]):
                if X[i, j] != 0:
                    line += f" {j+1}:{X[i, j]}"
            f.write(line + "\n")

def test_blinding():
    """Test blinding functionality"""
    print("Creating test data...")
    X, y = create_test_data()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as train_file:
        train_filename = train_file.name
        
    try:
        save_libsvm_format(X, y, train_filename)
        
        # Test without blinding
        print("Training without blinding...")
        cmd_no_blind = [
            './lightgbm',
            'train',
            '-d', train_filename,
            '-o', 'model_no_blind.txt',
            '--num_leaves', '31',
            '--num_iterations', '10',
            '--learning_rate', '0.1',
            '--blind_volume', '0.0'  # No blinding
        ]
        
        result = subprocess.run(cmd_no_blind, capture_output=True, text=True, cwd='.')
        if result.returncode != 0:
            print(f"Error training without blinding: {result.stderr}")
            return False
            
        # Test with blinding
        print("Training with blinding (blind_volume=0.1)...")
        cmd_with_blind = [
            './lightgbm',
            'train',
            '-d', train_filename,
            '-o', 'model_with_blind.txt',
            '--num_leaves', '31',
            '--num_iterations', '10', 
            '--learning_rate', '0.1',
            '--blind_volume', '0.1'  # 10% blinding
        ]
        
        result = subprocess.run(cmd_with_blind, capture_output=True, text=True, cwd='.')
        if result.returncode != 0:
            print(f"Error training with blinding: {result.stderr}")
            return False
            
        print("SUCCESS: Both training runs completed successfully!")
        print("Blinding feature implementation appears to be working.")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(train_filename):
            os.unlink(train_filename)
        for model_file in ['model_no_blind.txt', 'model_with_blind.txt']:
            if os.path.exists(model_file):
                os.unlink(model_file)

if __name__ == "__main__":
    success = test_blinding()
    exit(0 if success else 1)
