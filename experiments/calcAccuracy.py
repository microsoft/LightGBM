import numpy as np
from sklearn.metrics import accuracy_score

# File paths
predictions_file = 'LightGBM_predict_result.txt'
test_data_file = 'covtype.libsvm.binary.test'
config_file = 'train.conf'  # Path to train.conf

# Load predictions
with open(predictions_file, 'r') as f:
    predictions = np.array([float(line.strip()) for line in f])

# Convert probabilities to binary (threshold 0.5)
binary_predictions = (predictions >= 0.5).astype(int)

# Load test data (first column is the true label)
with open(test_data_file, 'r') as f:
    true_labels = np.array([int(line.split()[0]) for line in f])

# Calculate accuracy
accuracy = accuracy_score(true_labels, binary_predictions)

# Print accuracy
print(f'Accuracy: {accuracy:.4f}')

# Function to read max_depth and num_trees from the train.conf file, handling spaces and multiple values
def extract_train_config(config_file):
    max_depth = None
    num_trees = None

    with open(config_file, 'r') as f:
        for line in f:
            # Remove surrounding spaces and check for '='
            line = line.strip()
            if '=' in line:
                key_value = line.split('=')
                key = key_value[0].strip()  # Remove spaces around the key
                value = key_value[1].strip().split()[0]  # Take the first value after '=' and strip spaces

                if key == 'max_depth':
                    max_depth = int(value)
                if key == 'num_trees':
                    num_trees = int(value)

    return max_depth, num_trees

# Get config information from train.conf
max_depth, num_trees = extract_train_config(config_file)

# Print max_depth and num_trees
print(f'Max Depth: {max_depth}')
print(f'Number of Trees: {num_trees}')
