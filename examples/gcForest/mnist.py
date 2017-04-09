import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

n_row, n_col = 28, 28

X_train, y_train = load_svmlight_file('mnist.scale', n_features=n_row * n_col)
X_test, y_test = load_svmlight_file('mnist.scale.t', n_features=n_row * n_col)

X_train = X_train.toarray()
X_test = X_test.toarray()

cf = lgb.CascadeForest(num_forest=2, num_tree=100)
classVector = None
for i in range(2):
    X_train, classVector = cf.train(X_train, y_train, classVector=classVector)
    y_pred = cf.predict(X_test)
    print('accuracy_score:', accuracy_score(y_pred, y_test))
