Python Package Example
=====================
Here is an example for LightGBM to use python package.

***You should install lightgbm (both c++ and python verion) first.***

For the installation, check the wiki [here](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide).

You also need scikit-learn, pandas and matplotlib (only for plot example) to run the examples, but they are not required for the package itself. You can install them with pip:
```
pip install scikit-learn pandas matplotlib -U
```

Now you can run examples in this folder, for example:
```
python simple_example.py
```
Examples including:
- [simple_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py)
    - Construct Dataset
    - Basic train and predict
    - Eval during training 
    - Early stopping
    - Save model to file
    - Dump model to json format
    - Feature importances
- [sklearn_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py)
    - Basic train and predict with sklearn interface
    - Feature importances with sklearn interface
- [advanced_example.py](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py)
    - Set feature names
    - Directly use categorical features without one-hot encoding
    - Load model file to continue training
    - Change learning rates during training
    - Self-defined objective function
    - Self-defined eval metric
    - Callback function