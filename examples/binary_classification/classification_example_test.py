import unittest
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

class TestLightGBMCustomLoss(unittest.TestCase):

    def setUp(self):
        # Load the dataset
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target

        # Manually shuffle and split the data
        np.random.seed(42)
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        split_ratio = 0.7
        split_index = int(self.X.shape[0] * split_ratio)

        # Split the data
        self.X_train = self.X[indices[:split_index]]
        self.y_train = self.y[indices[:split_index]]
        self.X_test = self.X[indices[split_index:]]
        self.y_test = self.y[indices[split_index:]]

        # Create the LightGBM dataset
        self.train_data = lgb.Dataset(self.X_train, label=self.y_train)

        # Set model parameters
        self.params = {
            'boosting_type': 'gbdt',
            'objective': self.custom_logistic_obj,  # Use the custom cost function
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': 0
        }

    @staticmethod
    def custom_logistic_obj(preds, dtrain):
        """Custom logistic loss function"""
        labels = dtrain.get_label()
        
        # Compute the gradients and hessians
        preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid to get probabilities
        grad = preds - labels  # Gradient (first derivative)
        hess = preds * (1.0 - preds)  # Hessian (second derivative)
        
        # Debugging prints
        print("Preds:", preds)
        print("Grad:", grad)
        print("Hess:", hess)
        
        return grad, hess

    def test_model_training(self):
        """Test that the model trains without errors and outputs predictions."""
        # Train the model
        lgb_model = lgb.train(self.params, self.train_data, num_boost_round=100)
        
        # Predict on test data
        y_pred = lgb_model.predict(self.X_test)
        
        # Ensure predictions have correct shape
        self.assertEqual(y_pred.shape, (self.X_test.shape[0],), "Prediction shape mismatch")
        
        # Ensure predictions are probabilities (between 0 and 1) using the sigmoid function
        y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred))  # Apply sigmoid
        self.assertTrue(np.all((y_pred_probs >= 0) & (y_pred_probs <= 1)), "Predictions are not probabilities")

    def test_classification_performance(self):
        """Test that the classification performance is acceptable."""
        # Train the model
        lgb_model = lgb.train(self.params, self.train_data, num_boost_round=100)
        
        # Predict on test data
        y_pred = lgb_model.predict(self.X_test)
        
        # Convert probabilities to class labels
        y_pred_class = (y_pred > 0.5).astype(int)
        
        # Check classification report metrics
        report = metrics.classification_report(self.y_test, y_pred_class, output_dict=True)
        
        # Ensure precision, recall, and f1-score are reasonable
        self.assertGreater(report['1']['precision'], 0.8, "Precision is too low")
        self.assertGreater(report['1']['recall'], 0.8, "Recall is too low")
        self.assertGreater(report['1']['f1-score'], 0.8, "F1-score is too low")

if __name__ == '__main__':
    unittest.main()  