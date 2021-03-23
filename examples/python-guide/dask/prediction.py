import dask.array as da
from distributed import Client, LocalCluster
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

if __name__ == "__main__":
    print("loading data")

    X, y = make_regression(n_samples=1000, n_features=50)

    print("initializing a Dask cluster")

    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)

    print("created a Dask LocalCluster")

    print("distributing training data on the Dask cluster")

    dX = da.from_array(X, chunks=(100, 50))
    dy = da.from_array(y, chunks=(100,))

    print("beginning training")

    dask_model = lgb.DaskLGBMRegressor(n_estimators=10)
    dask_model.fit(dX, dy)
    assert dask_model.fitted_

    print("done training")

    print("predicting on the training data")

    preds = dask_model.predict(dX)

    # the code below uses sklearn.metrics, but this requires pulling all of the
    # predictions and target values back from workers to the client
    #
    # for larger datasets, consider the metrics from dask-ml instead
    # https://ml.dask.org/modules/api.html#dask-ml-metrics-metrics
    print("computing MSE")

    preds_local = preds.compute()
    actuals_local = dy.compute()
    mse = mean_squared_error(actuals_local, preds_local)

    print(f"MSE: {mse}")
