import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# model = lgb.Booster(model_str = './examples/binary_classification/LightGBM_model_tiny.txt')
model = lgb.Booster(model_file = './examples/binary_classification/LightGBM_model_tiny.txt')

lgb.plot_tree(model, tree_index=1, show_info=['internal_count', 'leaf_count', 'data_percentage'], orientation='vertical')
plt.show()