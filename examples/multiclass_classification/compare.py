import pandas as pd
pred1 = pd.read_csv('LightGBM_predict_result.txt', header=None, sep='\t').idxmax(1)
train = pd.read_csv('multiclass.train', header=None, sep='\t')
test = pd.read_csv('multiclass.test', header=None, sep='\t')


import xgboost as xgb
params = {
	'eta': 0.1,
	'objective': 'multi:softmax',
	'eval_metric': 'mlogloss',
	'num_class': 5,
	'silent': 1,
	'seed':42
}

ytrain, ytest = train[0], test[0]
Xtrain, Xtest = train.drop(0, axis=1), test.drop(0, axis=1)

from sklearn.metrics import accuracy_score
print 'accuracy for lightgbm:', accuracy_score(ytest, pred1)

xgbtrain = xgb.DMatrix(Xtrain, ytrain)
xgbtest = xgb.DMatrix(Xtest)

gbm = xgb.train(params, xgbtrain, 100)
pred2 = gbm.predict(xgbtest)

print 'accuracy for xgboost:', accuracy_score(ytest, pred2)