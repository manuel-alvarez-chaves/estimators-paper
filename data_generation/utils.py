import os
import h5py
import logging
import numpy as np


def get_logger(path_to_file):
    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(path_to_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Standard output
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # Logger instance
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_database(path_to_file):
    if not os.path.exists(path_to_file):
        h5py.File(path_to_file, "w")
    return None


def get_samples(func, limits, n_samples, seed=None, **kwargs):
    rng = np.random.default_rng(seed)

    ids = []
    acc_rate = 1.0
    while len(ids) < n_samples:
        d = len(limits)
        f = np.array(limits)[:, 0]  # floor
        s = np.array(limits)[:, 1] - f  # scale
        r = rng.uniform(size=(int(n_samples / acc_rate), d))
        r = f + s * r

        F = func(*(np.hsplit(r, d)), **kwargs).flatten()
        G = 1 / np.prod(s)
        M = F.max() / G

        U = rng.uniform(0.0, 1.0, size=F.shape)
        ids = np.argwhere(U < F / (M * G)).flatten()
        acc_rate = acc_rate * (len(ids) / n_samples)

    samples = r[ids][:n_samples, :]
    return samples


def save_to_file(path_to_file, case_1, case_2, dist_params, sample_sizes, seeds, data):
    with h5py.File(path_to_file, "r+") as f:
        try:
            f.create_group(case_1)
        except:
            pass
        try:
            f[case_1].create_group(case_2)
        except:
            pass
        g = f[case_1][case_2]
        g.attrs["hyper_params"] = str(dist_params)
        g.attrs["sample_sizes"] = sample_sizes
        g.attrs["seeds"] = seeds
        for k1, v1 in data.items():
            gg = g.create_group(str(k1))
            for k2, v2 in v1.items():
                gg.create_dataset(str(k2), data=v2)