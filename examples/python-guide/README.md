Python Package Examples
=======================

Here is an example for LightGBM to use Python-package.

You should install LightGBM [Python-package](https://github.com/Microsoft/LightGBM/tree/master/python-package) first.

You also need scikit-learn, pandas and matplotlib (only for plot example) to run the examples, but they are not required for the package itself. You can install them with pip:

```
pip install scikit-learn pandas matplotlib -U
```

Now you can run examples in this folder, for example:

```
python simple_example.py
```

Examples include:

- [simple_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py)
    - Construct Dataset
    - Basic train and predict
    - Eval during training 
    - Early stopping
    - Save model to file
- [sklearn_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py)
    - Basic train and predict with sklearn interface
    - Feature importances with sklearn interface
- [advanced_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py)
    - Set feature names
    - Directly use categorical features without one-hot encoding
    - Dump model to json format
    - Get feature importances
    - Get feature names
    - Load model to predict
    - Dump and load model with pickle
    - Load model file to continue training
    - Change learning rates during training
    - Self-defined objective function
    - Self-defined eval metric
    - Callback function
