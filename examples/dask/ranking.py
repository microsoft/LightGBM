import dask.array as da
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file
from distributed import Client, LocalCluster

X, y = load_svmlight_file("../lambdarank/rank.train")
group = np.loadtxt("../lambdarank/rank.train.query")

cluster = LocalCluster(n_workers=2)
client = Client(cluster)

# split training data into two partitions
rows_in_part1 = int(np.sum(group[:100]))
num_features = X.shape[1]

dX = da.from_array(
  x=X,
  chunks=[
    (rows_in_part1, num_features),
    (X.shape[0] - rows_in_part1, num_features)
  ]
)

dy = da.from_array(y)
dg = da.from_array(group)

dask_model = lgb.DaskLGBMRanker()
dask_model.fit(dX, dy, group=dg)
