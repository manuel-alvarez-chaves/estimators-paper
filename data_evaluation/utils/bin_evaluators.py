import time
import h5py
import numpy as np

from utils.base_evaluator import Evaluator
from unite_toolbox.bin_estimators import estimate_ideal_bins, calc_qs_entropy
from unite_toolbox.bin_estimators import calc_bin_entropy, calc_uniform_bin_entropy, calc_bin_mutual_information, calc_bin_kld

class Evaluator_BIN(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate_entropy(self, experiment: str, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    nbins = estimate_ideal_bins(data, counts=False)
                    for idx, hp in enumerate([hp for hp in self.hyper_params if hp != "qs"]):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = sum(list(calc_bin_entropy(data, edges=nbins[hp])))
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = sum(list(calc_bin_entropy(data, edges=nbins[hp])))
                        res[idx, idy, idz] = estimate
                    if "qs" in self.hyper_params and n == 10_000:
                        start_time = time.perf_counter()
                        estimate = calc_qs_entropy(data.flatten())
                        text = (
                            f"({experiment.upper()}, qs, {n}, {s}) "
                            f"- Time: {time.perf_counter() - start_time:.5f} s "
                            f"- Est.: {estimate:.3f} nats"
                        )
                        self.logger.info(text)
                        res[-1, idy, idz] = estimate
                    elif "qs" in self.hyper_params: 
                        res[-1, idy, idz] = calc_qs_entropy(data.flatten())
        self.results = res

    def evaluate_uniform_entropy(self, experiment: str, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    nbins = estimate_ideal_bins(data, counts=False)
                    for idx, hp in enumerate([hp for hp in self.hyper_params if hp != "qs"]):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = calc_uniform_bin_entropy(data, edges=nbins[hp])[0]
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = calc_uniform_bin_entropy(data, edges=nbins[hp])[0]
                        res[idx, idy, idz] = estimate
                    if "qs" in self.hyper_params and n == 10_000:
                        start_time = time.perf_counter()
                        estimate = calc_qs_entropy(data.flatten())
                        text = (
                            f"({experiment.upper()}, qs, {n}, {s}) "
                            f"- Time: {time.perf_counter() - start_time:.5f} s "
                            f"- Est.: {estimate:.3f} nats"
                        )
                        self.logger.info(text)
                        res[-1, idy, idz] = estimate
                    elif "qs" in self.hyper_params: 
                        res[-1, idy, idz] = calc_qs_entropy(data.flatten())
        self.results = res

    def evaluate_mutual_information(self, experiment: str, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    nbins = estimate_ideal_bins(data, counts=False)
                    for idx, hp in enumerate(self.hyper_params):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = calc_bin_mutual_information(data, nbins[hp])
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = calc_bin_mutual_information(data, nbins[hp])
                        res[idx, idy, idz] = estimate
        self.results = res

    def evaluate_uniform_mutual_information(self, experiment: str, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    nbins = estimate_ideal_bins(data, counts=False)
                    for idx, hp in enumerate(self.hyper_params):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate_xy = calc_uniform_bin_entropy(data, nbins[hp])
                            estimate_x = calc_uniform_bin_entropy(data[:, :-1], nbins[hp][:-1])
                            estimate_y = calc_uniform_bin_entropy(data[:, -1].reshape(-1, 1), [nbins[hp][-1]])
                            estimate = max(0.0, estimate_x[0] + estimate_y[0] - estimate_xy[0])
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate_xy = calc_uniform_bin_entropy(data, nbins[hp])
                            estimate_x = calc_uniform_bin_entropy(data[:, :-1], nbins[hp][:-1])
                            estimate_y = calc_uniform_bin_entropy(data[:, -1].reshape(-1, 1), [nbins[hp][-1]])
                            estimate = max(0.0, estimate_x[0] + estimate_y[0] - estimate_xy[0])
                        res[idx, idy, idz] = estimate
        self.results = res

    def evaluate_kld(self, experiment: str, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset1 = f[experiment]["p"]
            dset2 = f[experiment]["q"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    f = np.array(dset1[str(s)][str(n)])
                    g = np.array(dset2[str(s)][str(n)])
                    nbins = estimate_ideal_bins(g, counts=False)
                    for idx, hp in enumerate(self.hyper_params):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = calc_bin_kld(f, g, nbins[hp])
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = calc_bin_kld(f, g, nbins[hp])
                        res[idx, idy, idz] = estimate
        self.results = res