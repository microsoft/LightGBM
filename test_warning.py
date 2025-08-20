import subprocess

# Test with constant data to trigger the warning
constant_data = """1.0 1:1.0 2:1.0 3:1.0
1.0 1:1.0 2:1.0 3:1.0
1.0 1:1.0 2:1.0 3:1.0"""

with open('constant.txt', 'w') as f:
    f.write(constant_data)

print("Testing with constant features (should show blinding disabled warning):")
result = subprocess.run([
    './lightgbm', 'task=train', 'data=constant.txt', 
    'output_model=model.txt', 'num_iterations=1',
    'blind_volume=0.5'  # High blinding volume
], capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print(f"Return code: {result.returncode}")

import os
for f in ['constant.txt', 'model.txt']:
    if os.path.exists(f):
        os.unlink(f)
