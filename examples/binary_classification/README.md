Binary Classification Example
=============================

Here is an example for LightGBM to run binary classification task.

> **Note**
>
> Follow the [Installation Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) to make `lightgbm` binary available.

Training
--------

Run the following command in this folder:

```bash
lightgbm config=train.conf
```

> **Note**
>
> Use `train_linear.conf` for [fit piecewise linear gradient boosting tree](https://lightgbm.readthedocs.io/en/latest/Parameters.html#linear_tree).

Prediction
----------

> **Note**
>
> Finish the [training](#training) step first.

Run the following command in this folder:

```bash
lightgbm config=predict.conf
```
