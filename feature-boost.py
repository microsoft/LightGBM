from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest


class HDFSequence(lgb.Sequence):
    def __init__(self, hdf_dataset, batch_size):
        """
        Construct a sequence object from HDF5 with required interface.

        Parameters
        ----------
        hdf_dataset : h5py.Dataset
            Dataset in HDF5 file.
        batch_size : int
            Size of a batch. When reading data to construct lightgbm Dataset, each read reads batch_size rows.
        """
        self.data = hdf_dataset
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def auto_optimize_batch_size(df, max_batch_size=1024):
    """
    AI-based optimization to dynamically determine the best batch size.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    max_batch_size : int
        Maximum batch size limit.

    Returns
    -------
    int
        Optimized batch size.
    """
    optimal_batch_size = min(max(len(df) // 10, 32), max_batch_size)
    print(f"AI-optimized batch size: {optimal_batch_size}")
    return optimal_batch_size


def detect_data_anomalies(df):
    """
    AI-based anomaly detection in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Cleaned data with anomalies removed.
    """
    print("Running AI-based anomaly detection...")
    model = IsolationForest(contamination=0.01, random_state=42)  # Added a fixed seed for reproducibility
    outliers = model.fit_predict(df)
    cleaned_df = df[outliers == 1]
    num_anomalies = len(df) - len(cleaned_df)
    print(f"Detected and removed {num_anomalies} anomalies.")
    return cleaned_df


def intelligent_chunking_strategy(df, default_batch_size):
    """
    AI-based intelligent chunking strategy for HDF5 storage.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    default_batch_size : int
        Default batch size.

    Returns
    -------
    int
        Optimized chunk size.
    """
    avg_row_size = df.memory_usage(deep=True).sum() / len(df)
    chunk_size = min(max(int(default_batch_size / avg_row_size), 1), default_batch_size)
    print(f"AI-optimized chunk size: {chunk_size}")
    return chunk_size


def save2hdf(input_data, fname, batch_size):
    """Store numpy array to HDF5 file."""
    with h5py.File(fname, "w") as f:
        for name, data in input_data.items():
            nrow, ncol = data.shape
            if ncol == 1:
                chunk = (nrow,)
                data = data.values.flatten()
            else:
                chunk = (batch_size, ncol)
            f.create_dataset(name, data=data, chunks=chunk, compression="lzf")


def generate_hdf(input_fname, output_basename, batch_size):
    # Load and clean data
    df = pd.read_csv(input_fname, header=None, sep="\t")
    df = detect_data_anomalies(df)

    # Auto-optimize batch size
    optimized_batch_size = auto_optimize_batch_size(df)  # Introduced a new variable for the optimized batch size

    # Split data into two parts for demonstration
    mid = len(df) // 2
    df1 = df.iloc[:mid]
    df2 = df.iloc[mid:]

    # Determine chunk size
    chunk_size = intelligent_chunking_strategy(df1, optimized_batch_size)

    # Save to 2 HDF5 files
    fname1 = f"{output_basename}1.h5"
    fname2 = f"{output_basename}2.h5"
    save2hdf({"Y": df1.iloc[:, :1], "X": df1.iloc[:, 1:]}, fname1, chunk_size)
    save2hdf({"Y": df2.iloc[:, :1], "X": df2.iloc[:, 1:]}, fname2, chunk_size)

    # Data summary report
    print(f"Data summary: {df.describe()}")

    return [fname1, fname2]


def create_dataset_from_multiple_hdf(input_flist, batch_size):
    data = []
    ylist = []
    for f in input_flist:
        f = h5py.File(f, "r")
        data.append(HDFSequence(f["X"], batch_size))
        ylist.append(f["Y"][:])

    params = {
        "bin_construct_sample_cnt": 200000,
        "max_bin": 255,
    }
    y = np.concatenate(ylist)
    dataset = lgb.Dataset(data, label=y, params=params)
    dataset.save_binary("regression.train.from_hdf.bin")


def main():
    batch_size = 64
    output_basename = "regression"
    hdf_files = generate_hdf(
        str(Path(__file__).absolute().parents[1] / "regression" / "regression.train"), output_basename, batch_size
    )

    create_dataset_from_multiple_hdf(hdf_files, batch_size=batch_size)


if __name__ == "__main__":
    main()
