import sys

import numpy as np


def gen_csv(fname, nrow, ncol):
    nrow = int(nrow)
    ncol = int(ncol)

    arr = np.random.random(nrow * ncol) * 5
    arr = arr.reshape((nrow, ncol))
    np.savetxt(fname, arr, fmt='%.19f', delimiter=',')


if __name__ == '__main__':
    import argh
    argh.dispatch_command(gen_csv)

