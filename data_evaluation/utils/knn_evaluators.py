import time
import h5py
import numpy as np

from utils.base_evaluator import Evaluator

from unite_toolbox.knn_estimators import calc_knn_mutual_information, calc_knn_kld

class Evaluator_KNN(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate_mutual_information(self, experiment: str, hp_to_time: int):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    _, m = data.shape
                    x = data[:, :m-1].reshape(-1, m-1)
                    y = data[:, m-1].reshape(-1, 1)
                    for idx, hp in enumerate(self.hyper_params):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = calc_knn_mutual_information(x, y, hp)
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = calc_knn_mutual_information(x, y, hp)
                        res[idx, idy, idz] = estimate
        self.results = res

    def evaluate_kld(self, experiment: str, hp_to_time: int):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset1 = f[experiment]["p"]
            dset2 = f[experiment]["q"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    f = np.array(dset1[str(s)][str(n)])
                    g = np.array(dset2[str(s)][str(n)])
                    for idx, hp in enumerate(self.hyper_params):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = calc_knn_kld(f, g, hp)
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = calc_knn_kld(f, g, hp)
                        res[idx, idy, idz] = estimate
        self.results = res