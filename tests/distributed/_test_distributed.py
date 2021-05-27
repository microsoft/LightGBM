import copy
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generator, List

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score

import lightgbm as lgb


def _generate_n_ports(n: int) -> Generator[int, None, None]:
    return (lgb.dask._find_random_open_port() for _ in range(n))


def _write_dict(d: Dict, file: io.TextIOWrapper) -> None:
    for k, v in d.items():
        file.write(f'{k} = {v}\n')


def create_data(task: str, n_samples: int = 1_000) -> np.ndarray:
    """Create the appropiate data for the task.

    The data is returned as a numpy array with the label as the first column.
    """
    if task == 'binary-classification':
        centers = [[-4, -4], [4, 4]]
        X, y = make_blobs(n_samples, centers=centers, random_state=42)
    elif task == 'regression':
        X, y = make_regression(n_samples, n_features=4, n_informative=2, random_state=42)
    dataset = np.hstack([y.reshape(-1, 1), X])
    return dataset


class DistributedMockup:
    """Simulate distributed training."""

    default_train_config = {
        'task': 'train',
        'pre_partition': True,
        'machine_list_file': 'mlist.txt',
        'tree_learner': 'data',
        'force_row_wise': True,
        'verbose': 0,
        'num_boost_round': 20,
        'num_leaves': 15,
        'num_threads': 2,
    }

    default_predict_config = {
        'task': 'predict',
        'data': 'train.txt',
        'input_model': 'model0.txt',
        'output_result': 'predictions.txt',
    }

    def worker_train(self, i: int) -> subprocess.CompletedProcess:
        """Start the training process on the `i`-th worker.

        If this is the first worker, its logs are piped to stdout.
        """
        cmd = f'./lightgbm config=train{i}.conf'.split()
        return subprocess.run(cmd)

    def _set_ports(self) -> None:
        """Randomly assign a port for training to each worker and save all ports to mlist.txt."""
        ports = set(_generate_n_ports(self.n_workers))
        i = 0
        max_tries = 100
        while i < max_tries and len(ports) < self.n_workers:
            n_ports_left = self.n_workers - len(ports)
            candidates = _generate_n_ports(n_ports_left)
            ports.update(candidates)
            i += 1
        if i == max_tries:
            raise RuntimeError('Unable to find non-colliding ports.')
        self.listen_ports = list(ports)
        with open('mlist.txt', 'wt') as f:
            for port in self.listen_ports:
                f.write(f'127.0.0.1 {port}\n')

    def _write_data(self, partitions: List[np.ndarray]) -> None:
        """Write all training data as train.txt and each training partition as train{i}.txt."""
        all_data = np.vstack(partitions)
        np.savetxt('train.txt', all_data, delimiter=',')
        for i, partition in enumerate(partitions):
            np.savetxt(f'train{i}.txt', partition, delimiter=',')

    def fit(self, partitions: List[np.ndarray], train_config: Dict = {}) -> None:
        """Run the distributed training process on a single machine.

        For each worker i:
            1. The i-th partition is saved as train{i}.txt.
            2. A random port is assigned for training.
            3. A configuration file train{i}.conf is created.
            4. The lightgbm binary is called with config=train{i}.conf in another thread.
            5. The trained model is saved as model{i}.txt. Each model file only differs in data and local_listen_port.
        The whole training set is saved as train.txt and the logs from the first worker are piped to stdout.
        """
        self.train_config = copy.deepcopy(self.default_train_config)
        self.train_config.update(train_config)
        self.n_workers = self.train_config['num_machines']
        self._set_ports()
        self._write_data(partitions)
        self.label_ = np.hstack([partition[:, 0] for partition in partitions])
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i in range(self.n_workers):
                self.write_train_config(i)
                train_future = executor.submit(self.worker_train, i)
                futures.append(train_future)
            results = [f.result() for f in futures]
        for result in results:
            if result.returncode != 0:
                raise RuntimeError

    def predict(self, predict_config: Dict = {}) -> np.ndarray:
        """Compute the predictions using the model created in the fit step.

        model0.txt is used to predict the training set train.txt using predict.conf.
        The predictions are saved as predictions.txt and are then loaded to return them as a numpy array.
        The logs are piped to stdout.
        """
        self.predict_config = copy.deepcopy(self.default_predict_config)
        self.predict_config.update(predict_config)
        with open('predict.conf', 'wt') as file:
            _write_dict(self.predict_config, file)
        cmd = './lightgbm config=predict.conf'.split()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError
        y_pred = np.loadtxt('predictions.txt')
        return y_pred

    def write_train_config(self, i: int) -> None:
        """Create a file train{i}.conf with the required configuration to train.

        Each worker gets a different port and piece of the data, the rest are the
        model parameters contained in `self.config`.
        """
        with open(f'train{i}.conf', 'wt') as file:
            file.write(f'output_model = model{i}.txt\n')
            file.write(f'local_listen_port = {self.listen_ports[i]}\n')
            file.write(f'data = train{i}.txt\n')
            _write_dict(self.train_config, file)


def test_classifier():
    """Test the classification task."""
    num_machines = 2
    data = create_data(task='binary-classification')
    partitions = np.array_split(data, num_machines)
    train_params = {
        'objective': 'binary',
        'num_machines': num_machines,
    }
    clf = DistributedMockup()
    clf.fit(partitions, train_params)
    y_probas = clf.predict()
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.


def test_regressor():
    """Test the regression task."""
    num_machines = 2
    data = create_data(task='regression')
    partitions = np.array_split(data, num_machines)
    train_params = {
        'objective': 'regression',
        'num_machines': num_machines,
    }
    reg = DistributedMockup()
    reg.fit(partitions, train_params)
    y_pred = reg.predict()
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.2, atol=50.)
