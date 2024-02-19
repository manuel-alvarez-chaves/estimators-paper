import os, time
import h5py
import numpy as np

class Evaluator():
    """Generic class to evaluate estimators."""
    def __init__(self):
        self.data_path = None
        self.out_path = None
        self.logger = None
        self.quantity = None
        self.hyper_params = None
        self.sample_sizes = None
        self.seeds = None
        self.results = None
        
    def create_database(self):
        if not os.path.exists(self.out_path):
            with h5py.File(self.out_path, "w") as f:
                pass

    def create_group(self):
        with h5py.File(self.out_path, "r+") as f:
            try:
                f.create_group(self.quantity)
            except:
                pass

    def write_single_to_hdf5(self, experiment, truth):
        with h5py.File(self.data_path, "r") as f:
            params = f[experiment]["p"].attrs["hyper_params"]

        try:
            with h5py.File(self.out_path, "r+") as f:
                g = f[self.quantity]
                gg = g.create_group(experiment)
                gg.attrs["hyper_params"] = self.hyper_params
                gg.attrs["dist_params"] = str(params)
                gg.attrs[self.quantity] = truth
                gg.attrs["sample_sizes"] = self.sample_sizes
                gg.attrs["seeds"] = self.seeds
                gg.create_dataset("results", data = self.results, chunks=True, maxshape=(self.results.shape[0], self.results.shape[1], None))
        except:
            pass

    def write_double_to_hdf5(self, experiment, name_to_write, truth):
        with h5py.File(self.data_path, "r") as f:
            params1 = f[experiment]["p"].attrs["hyper_params"]
            params2 = f[experiment]["q"].attrs["hyper_params"]

        try:
            with h5py.File(self.out_path, "r+") as f:
                g = f[self.quantity]
                gg = g.create_group(name_to_write)
                gg.attrs["hyper_params"] = self.hyper_params
                gg.attrs["dist1_params"] = str(params1)
                gg.attrs["dist2_params"] = str(params2)
                gg.attrs[self.quantity] = truth
                gg.attrs["sample_sizes"] = self.sample_sizes
                gg.attrs["seeds"] = self.seeds
                gg.create_dataset("results", data = self.results)
        except:
            pass

    def append_to_hdf5(self, name_to_access, data_to_append):
        with h5py.File(self.out_path, "a") as f:
            # Attributes
            seeds_in_dset = f[self.quantity][name_to_access].attrs["seeds"]
            del f[self.quantity][name_to_access].attrs["seeds"]
            f[self.quantity][name_to_access].attrs["seeds"] = np.append(seeds_in_dset, self.seeds)

            # Data
            dset = f[self.quantity][name_to_access]
            dset = f[self.quantity][name_to_access]["results"]
            dset.resize(dset.shape[2] + data_to_append.shape[2], axis=2)
            dset[:, :, -data_to_append.shape[2]:] = data_to_append

    def evaluate(self, experiment: str, estimator: callable, hp_to_time: str):
        res = np.zeros(shape=(len(self.hyper_params), len(self.sample_sizes), len(self.seeds)))
        with h5py.File(self.data_path, "r") as f:
            dset = f[experiment]["p"]
            for idz, s in enumerate(self.seeds):
                for idy, n in enumerate(self.sample_sizes):
                    data = np.array(dset[str(s)][str(n)])
                    for idx, hp in enumerate([hp for hp in self.hyper_params if hp != "qs"]):
                        if n == 10_000 and hp == hp_to_time: # Time for 10 000 samples
                            start_time = time.perf_counter()
                            estimate = estimator(data, hp)
                            text = (
                                f"({experiment.upper()}, {hp}, {n}, {s}) "
                                f"- Time: {time.perf_counter() - start_time:.5f} s "
                                f"- Est.: {estimate:.3f} nats"
                            )
                            self.logger.info(text)
                        else:
                            estimate = estimator(data, hp)
                        res[idx, idy, idz] = estimate
        self.results = res