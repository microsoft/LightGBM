import copy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import accuracy_score

import lightgbm as lgb


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


def run_and_log(cmd: List[str]) -> None:
    """Run `cmd` in another process and pipe its logs to this process' stdout."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert process.stdout is not None

    def stdout_stream():
        return process.stdout.read(1)

    for c in iter(stdout_stream, b''):
        sys.stdout.buffer.write(c)


class DistributedMockup:
    """Simulate distributed training."""

    default_config = {
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

    def __init__(self, config: Dict = {}):
        self.config = copy.deepcopy(self.default_config)
        self.config.update(config)
        self.n_workers = self.config['num_machines']

    def worker_train(self, i: int) -> None:
        """Start the training process on the `i`-th worker.

        If this is the first worker, its logs are piped to stdout.
        """
        cmd = f'./lightgbm config=train{i}.conf'.split()
        if i == 0:
            return run_and_log(cmd)
        subprocess.run(cmd)

    def _set_ports(self) -> None:
        """Randomly assign a port for training to each worker and save all ports to mlist.txt."""
        self.listen_ports = [lgb.dask._find_random_open_port() for _ in range(self.n_workers)]
        with open('mlist.txt', 'wt') as f:
            for port in self.listen_ports:
                f.write(f'127.0.0.1 {port}\n')

    def _write_data(self, partitions: List[np.ndarray]) -> None:
        """Write all training data as train.txt and each training partition as train{i}.txt."""
        all_data = np.vstack(partitions)
        np.savetxt('train.txt', all_data, delimiter=',')
        for i, partition in enumerate(partitions):
            np.savetxt(f'train{i}.txt', partition, delimiter=',')

    def fit(self, partitions: List[np.ndarray]) -> None:
        """Run the distributed training process on a single machine.

        For each worker i:
            1. The i-th partition is saved as train{i}.txt
            2. A random port is assigned for training.
            3. A configuration file train{i}.conf is created.
            4. The lightgbm binary is called with config=train{i}.conf in another thread.
            5. The trained model is saved as model{i}.txt. Each model file only differs in data and local_listen_port.
        The whole training set is saved as train.txt and the logs from the first worker are piped to stdout.
        """
        self._write_data(partitions)
        self.label_ = np.hstack([partition[:, 0] for partition in partitions])
        self._set_ports()
        futures = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i in range(self.n_workers):
                self.write_train_config(i)
                futures.append(executor.submit(self.worker_train, i))
            _ = [f.result() for f in futures]

    def predict(self) -> np.ndarray:
        """Compute the predictions using the model created in the fit step.

        model0.txt is used to predict the training set train.txt using predict.conf.
        The predictions are saved as predictions.txt and are then loaded to return them as a numpy array.
        The logs are piped to stdout.
        """
        cmd = './lightgbm config=predict.conf'.split()
        run_and_log(cmd)
        y_pred = np.loadtxt('predictions.txt')
        return y_pred

    def write_train_config(self, i: int) -> None:
        """Create a file train{i}.conf with the required configuration to train.

        Each worker gets a different port and piece of the data, the rest are the
        model parameters contained in `self.config`.
        """
        with open(f'train{i}.conf', 'wt') as f:
            f.write(f'output_model = model{i}.txt\n')
            f.write(f'local_listen_port = {self.listen_ports[i]}\n')
            f.write(f'data = train{i}.txt\n')
            for param, value in self.config.items():
                f.write(f'{param} = {value}\n')


def test_classifier():
    """Test the classification task."""
    num_machines = 2
    data = create_data(task='binary-classification')
    partitions = np.array_split(data, num_machines)
    params = {
        'objective': 'binary',
        'num_machines': num_machines,
    }
    clf = DistributedMockup(params)
    clf.fit(partitions)
    y_probas = clf.predict()
    y_pred = y_probas > 0.5
    assert accuracy_score(clf.label_, y_pred) == 1.


def test_regressor():
    """Test the regression task."""
    num_machines = 2
    data = create_data(task='regression')
    partitions = np.array_split(data, num_machines)
    params = {
        'objective': 'regression',
        'num_machines': num_machines,
    }
    reg = DistributedMockup(params)
    reg.fit(partitions)
    y_pred = reg.predict()
    np.testing.assert_allclose(y_pred, reg.label_, rtol=0.2, atol=50.)
