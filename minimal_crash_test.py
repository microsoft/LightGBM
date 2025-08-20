import subprocess

# Test with the exact same data but with blind_volume=0.0 vs 0.01
constant_data = """1.0 1:1.0 2:1.0 3:1.0
1.0 1:1.0 2:1.0 3:1.0
1.0 1:1.0 2:1.0 3:1.0"""

with open('constant.txt', 'w') as f:
    f.write(constant_data)

print("Testing blind_volume=0.0 (should work):")
result1 = subprocess.run([
    './lightgbm', 'task=train', 'data=constant.txt', 
    'output_model=model1.txt', 'num_iterations=1',
    'blind_volume=0.0', 'verbosity=-1'
], capture_output=True)
print(f"Return code: {result1.returncode}")

print("\nTesting blind_volume=0.01 (crashes):")
result2 = subprocess.run([
    './lightgbm', 'task=train', 'data=constant.txt', 
    'output_model=model2.txt', 'num_iterations=1',
    'blind_volume=0.01', 'verbosity=-1'
], capture_output=True)
print(f"Return code: {result2.returncode}")

import os
for f in ['constant.txt', 'model1.txt', 'model2.txt']:
    if os.path.exists(f):
        os.unlink(f)
