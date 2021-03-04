import dask.array as da
import lightgbm as lgb
from distributed import Client, LocalCluster
from sklearn.datasets import make_blobs

print("loading data")
X, y = make_blobs(n_samples=1000, n_features=50, centers=3)

print("initializing a Dask cluster")
cluster = LocalCluster(n_workers=2)
client = Client(cluster)
print("created a Dask LocalCluster")

print("distributing training data on the Dask cluster")
dX = da.from_array(X, chunks=(100, 50))
dy = da.from_array(y, chunks=(100,))

print("beginning training")
dask_model = lgb.DaskLGBMClassifier()
dask_model.fit(dX, dy)
print("done training")
