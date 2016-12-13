# coding: utf-8
# pylint: disable = C0103, C0111, C0301, C0321, C0330, W0621
import inspect
import lightgbm as lgb

file_api = open('Python_API.md', 'w+')

def write_func(func, leftSpace=0):
    file_api.write('####' + func.__name__ + '('
        + ', '.join([
                v.name + ('=' + str(v.default) if v.default != v.empty else '')
                for _, v in inspect.signature(func).parameters.items() if v.name != 'self'
            ])
        + ')\n')
    if func.__doc__:
        for line in func.__doc__.splitlines():
            if line: file_api.write(line[leftSpace:])
            file_api.write('\n')
    file_api.write('\n')

def write_class(class_):
    file_api.write('###' + class_.__name__ + '\n')
    for name, members in sorted(class_.__dict__.items(), key=lambda x: x[0]):
        if name == '__init__' or not name.startswith('_'): write_func(members, leftSpace=4)

def write_module(name, members):
    file_api.write('##' + name + '\n----\n')
    for member in members:
        if inspect.isclass(member): write_class(member)
        else: write_func(member)

write_module('Basic Data Structure API', [
        lgb.Dataset,
        lgb.Booster
    ])
write_module('Training API', [
        lgb.train,
        lgb.cv
    ])
write_module('Scikit-learn API', [
        lgb.LGBMModel,
        lgb.LGBMClassifier,
        lgb.LGBMRegressor,
        lgb.LGBMRanker
    ])

file_api.close()
