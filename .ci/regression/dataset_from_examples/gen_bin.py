import sys
import lightgbm as lgb
import pandas as pd

data_path = sys.argv[1]

print('Loading data...')
# load or create your dataset
df_train = pd.read_csv('../../../examples/regression/regression.train', header=None, sep='\t', float_precision='round_trip')
df_test = pd.read_csv('../../../examples/regression/regression.test', header=None, sep='\t', float_precision='round_trip')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_train.save_binary(f"{data_path}/train.bin")
# lgb_eval.save_binary(f"{data_path}/valid.bin")
