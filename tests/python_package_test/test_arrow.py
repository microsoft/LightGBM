# coding: utf-8
import filecmp
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pyarrow as pa
import pytest

import lightgbm as lgb

# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #


def generate_simple_arrow_table() -> pa.Table:
    columns = [
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint8()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int8()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint16()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int16()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint64()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int64()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.float32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.float64()),
    ]
    return pa.Table.from_arrays(columns, names=[f"col_{i}" for i in range(len(columns))])


def generate_dummy_arrow_table() -> pa.Table:
    col1 = pa.chunked_array([[1, 2, 3], [4, 5]], type=pa.uint8())
    col2 = pa.chunked_array([[0.5, 0.6], [0.1, 0.8, 1.5]], type=pa.float32())
    return pa.Table.from_arrays([col1, col2], names=["a", "b"])


def generate_random_arrow_table(num_columns: int, num_datapoints: int, seed: int) -> pa.Table:
    columns = [generate_random_arrow_array(num_datapoints, seed + i) for i in range(num_columns)]
    names = [f"col_{i}" for i in range(num_columns)]
    return pa.Table.from_arrays(columns, names=names)


def generate_random_arrow_array(num_datapoints: int, seed: int) -> pa.ChunkedArray:
    generator = np.random.default_rng(seed)
    data = generator.standard_normal(num_datapoints)

    # Set random nulls
    indices = generator.choice(len(data), size=num_datapoints // 10)
    data[indices] = None

    # Split data into <=2 random chunks
    split_points = np.sort(generator.choice(np.arange(1, num_datapoints), 2, replace=False))
    split_points = np.concatenate([[0], split_points, [num_datapoints]])
    chunks = [data[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    # Turn chunks into array
    return pa.chunked_array([data], type=pa.float32())


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
    }


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- DATASET ------------------------------------------- #


@pytest.mark.parametrize(
    ("arrow_table_fn", "dataset_params"),
    [  # Use lambda functions here to minimize memory consumption
        (lambda: generate_simple_arrow_table(), dummy_dataset_params()),
        (lambda: generate_dummy_arrow_table(), dummy_dataset_params()),
        (lambda: generate_random_arrow_table(3, 1000, 42), {}),
        (lambda: generate_random_arrow_table(100, 10000, 43), {}),
    ],
)
def test_dataset_construct_fuzzy(
    tmp_path: Path, arrow_table_fn: Callable[[], pa.Table], dataset_params: Dict[str, Any]
):
    arrow_table = arrow_table_fn()

    arrow_dataset = lgb.Dataset(arrow_table, params=dataset_params)
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(arrow_table.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    arrow_dataset._dump_text(tmp_path / "arrow.txt")
    pandas_dataset._dump_text(tmp_path / "pandas.txt")
    assert filecmp.cmp(tmp_path / "arrow.txt", tmp_path / "pandas.txt")
