import time
import h5py
import numpy as np

from utils.base_evaluator import Evaluator

from unite_toolbox.kde_estimators import calc_kde_mutual_information, calc_kde_kld
from unite_toolbox.kde_estimators import calc_ikde_mutual_information, calc_ikde_kld

class Evaluator_KDE(Evaluator):
    def __init__(self):
        super().__init__()
    
    def evaluate_mutual_information(self, experiment: str, hp_to_time: str, integration: bool):
        # Switch Estimator
        if integration: estimator = calc_ikde_mutual_information
        else: estimator = calc_kde_mutual_information

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
                            estimate = estimator(x, y, hp)
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = estimator(x, y, hp)
                        res[idx, idy, idz] = estimate
        self.results = res

    def evaluate_kld(self, experiment: str, hp_to_time: str, integration: bool):
        # Switch Estimator
        if integration: estimator = calc_ikde_kld
        else: estimator = calc_kde_kld

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
                            estimate = estimator(f, g, hp)
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = estimator(f, g, hp)
                        res[idx, idy, idz] = estimate
        self.results = res