
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Manually shuffle and split the data
np.random.seed(42)

# Shuffle the indices
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Define the split ratio
split_ratio = 0.7
split_index = int(X.shape[0] * split_ratio)

# Split the data
X_train = X[indices[:split_index]]
y_train = y[indices[:split_index]]
X_test = X[indices[split_index:]]
y_test = y[indices[split_index:]]

# Define custom logistic loss function 
""" This explains how to create a custom cost function for binary classification in LightGBM. 
The model provides raw predictions (preds), which are transformed into probabilities using the sigmoid function. 
The true labels (labels) are compared with the predicted probabilities to calculate the gradient (grad), which is the difference between the two. The second derivative, or Hessian (hess), is derived from the logistic loss function. During training, LightGBM uses this custom objective function to adjust model parameters. 
After training, predictions are converted into class labels and evaluated using standard metrics by thresholding probabilities at 0.5. """
def custom_logistic_obj(preds, dtrain):
    """Custom logistic loss function"""
    labels = dtrain.get_label()
    
    # Apply sigmoid to raw predictions to get probabilities
    preds = 1.0 / (1.0 + np.exp(-preds))
    
    # Gradient (first derivative)
    grad = preds - labels
    
    # Hessian (second derivative)
    hess = preds * (1.0 - preds)
    
    return grad, hess

# Create the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set parameters
params = {
    'boosting_type': 'gbdt',
    'objective': custom_logistic_obj,  # Use the custom cost function
    'metric': 'binary_logloss',        # Standard metric for evaluation
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': 0
}

# Train the model using the custom logistic loss function
lgb_model = lgb.train(params, train_data, num_boost_round=100)

# Predict on the test data
y_pred = lgb_model.predict(X_test)

# Convert probabilities to class labels
y_pred_class = (y_pred > 0.5).astype(int)

# Evaluate the model
print(metrics.classification_report(y_test, y_pred_class))
