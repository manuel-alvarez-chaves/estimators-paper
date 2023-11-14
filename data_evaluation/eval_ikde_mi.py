# File/OS management
import os, sys, time
import h5py, ast

# Required modules
import numpy as np
from scipy import stats
from scipy.special import gamma, digamma
from scipy.integrate import nquad

from utils.tools import get_logger
from utils.kde_evaluators import Evaluator_KDE

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

os.chdir(sys.path[0]) # Set location of file to CWD

# Evaluator attributes
eval = Evaluator_KDE()
eval.data_path = "data/data.hdf5"
eval.out_path = "results/ikde.hdf5"
eval.logger = get_logger("results/ikde_mi.log")

eval.quantity = "MI"

eval.hyper_params = ["silverman"]
eval.sample_sizes = [100, 200, 500, 1_000, 5_000, 10_000, 25_000]
eval.seeds = range(1, 3)

# Create database (if not existing)
eval.create_database()
eval.create_group()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # BIVARIATE-NORMAL # # # # #

experiment = "bivariate-normal"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

cov = np.array(dist_params[0][1])
d = len(cov)
true_mi = 0.5 * np.log(cov[0, 0] * cov[-1, -1] / np.linalg.det(cov)) # Reference

start_time = time.perf_counter()
eval.evaluate_mutual_information(experiment, "silverman", True)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_mi)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True MI: {true_mi:.3f} nats")

# # # # # BIVARIATE-NORMAL-MIXTURE # # # # #

experiment = "bivariate-normal-mixture"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

def pdf_normal(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.norm(loc=l, scale=s).pdf(x) * w
    return y

def pdf_mnorm(x, y, params):
    z = 0.0
    for dist in params:
        l, s, w = dist
        z += stats.multivariate_normal(mean=l, cov=s).pdf(np.dstack((x, y))) * w
    return z

def mi_mnorm(x, y, params1):
    params_x = []
    params_y = []
    for dist in params1:
        params_x.append([dist[0][0], dist[1][0][0], dist[2]])
        params_y.append([dist[0][1], dist[1][1][1], dist[2]])

    pxy = pdf_mnorm(x, y, params1)
    px = pdf_normal(x, params_x)
    py = pdf_normal(y, params_y)
    
    return pxy * np.log(pxy / (px * py))


binorm_lims = [[-7, 7], [-7, 7]]

true_mi = nquad(mi_mnorm, binorm_lims, args=(dist_params,))[0] # Numerical Integration Result

start_time = time.perf_counter()
eval.evaluate_mutual_information(experiment, "silverman", True)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_mi)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True MI: {true_mi:.3f} nats")

# # # # # GAMMA-EXPONENTIAL # # # # #

experiment = "gexp"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

tetha = dist_params[0][0]
true_mi = digamma(tetha) - np.log(tetha) + (1 / tetha) # Reference

start_time = time.perf_counter()
eval.evaluate_mutual_information(experiment, "silverman", True)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_mi)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True MI: {true_mi:.3f} nats")