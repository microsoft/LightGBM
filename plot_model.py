import matplotlib.pyplot as plt
import lightgbm as lgb

model = lgb.Booster(model_file = './examples/binary_classification/LightGBM_model_tiny.txt')

# lgb.plot_tree(model, tree_index=3, show_info=['internal_count', 'leaf_count', 'data_percentage'], orientation='vertical')
num_trees = model.num_trees()
for i in range(2):
    ax = lgb.plot_tree(model, tree_index=i, figsize=(20, 10), show_info=['split_gain', 'internal_value', 'leaf_count'])
    plt.title(f'Tree {i}')
    plt.show()
plt.show()