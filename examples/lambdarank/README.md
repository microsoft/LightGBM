LambdaRank Example
==================

Here is an example for LightGBM to run lambdarank task.

***You should copy executable file to this folder first.***

Training
--------

Run the following command in this folder:

```
"./lightgbm" config=train.conf
```

Prediction
----------

You should finish training first.

Run the following command in this folder:

```
"./lightgbm" config=predict.conf
```
