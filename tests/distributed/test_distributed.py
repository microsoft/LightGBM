import copy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score


def create_data(task, n_samples=1_000):
    if task == 'binary-classification':
        centers = [[-4, -4], [4, 4]]
        X, y = make_blobs(n_samples, centers=centers, random_state=42)
    elif task == 'regression':
        X, y = make_regression(n_samples, n_features=4, n_informative=2, random_state=42)
    dataset = np.hstack((y[:, None], X))
    return dataset    


def run_and_log(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''): 
        sys.stdout.buffer.write(c)


class DistributedMockup:
    default_config = {
        'output_model': 'model.txt',
        'machine_list_file': 'mlist.txt',
        'tree_learner': 'data',
        'force_row_wise': True,
        'verbose': 0,
        'num_boost_round': 20,
        'num_leaves': 15,
        'num_threads': 2,
    }
    def __init__(self, config={}, n_workers=2):
        self.config = copy.deepcopy(self.default_config)
        self.config.update(config)
        self.config['num_machines'] = n_workers
        self.n_workers = n_workers

    def worker_train(self, i):
        cmd = f'./lightgbm config=train{i}.conf'.split()
        if i == 0:
            return run_and_log(cmd)
        subprocess.run(cmd)
            
    def _set_ports(self):
        self.listen_ports = [lgb.dask._find_random_open_port() for _ in range(self.n_workers)]
        with open('mlist.txt', 'wt') as f:
            for port in self.listen_ports:
                f.write(f'127.0.0.1 {port}\n')

    def _write_data(self, data):
        np.savetxt('train.txt', data, delimiter=',')
        for i, partition in enumerate(np.array_split(data, self.n_workers)):
            np.savetxt(f'train{i}.txt', partition, delimiter=',')

    def fit(self, data):
        self._write_data(data)
        self.label_ = data[:, 0]
        self._set_ports()
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i in range(self.n_workers):
                self.write_train_config(i)
                futures.append(executor.submit(self.worker_train, i))
            results = [f.result() for f in futures]
            
    def predict(self):
        cmd = './lightgbm config=predict.conf'.split()
        run_and_log(cmd)
        y_pred = np.loadtxt('predictions.txt')
        return y_pred

    def write_train_config(self, i):
        with open(f'train{i}.conf', 'wt') as f:
            f.write('task = train\n')
            f.write(f'local_listen_port = {self.listen_ports[i]}\n')
            f.write(f'data = train{i}.txt\n')
            for param, value in self.config.items():
                f.write(f'{param} = {value}\n')


def test_classifier():
    data = create_data(task='binary-classification')
    params = {
        'objective': 'binary',
    }
    clf = DistributedMockup(params)
    clf.fit(data)
    y_probas = clf.predict()
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.


def test_regressor():
    data = create_data(task='regression')
    params = {
        'objective': 'regression',
    }
    reg = DistributedMockup(params)
    reg.fit(data)
    y_pred = reg.predict()
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.5, atol=50.)