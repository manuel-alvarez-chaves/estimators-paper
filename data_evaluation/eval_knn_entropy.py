# File/OS management
import os, sys, time
import h5py, ast

# Required modules
import numpy as np
from scipy import stats
from scipy.special import gamma, digamma
from scipy.integrate import nquad

from utils.tools import get_logger
from utils.knn_evaluators import Evaluator_KNN
from unite_toolbox.knn_estimators import calc_knn_entropy

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

os.chdir(sys.path[0]) # Set location of file to CWD

# Evaluator attributes
eval = Evaluator_KNN()
eval.data_path = "data/data.hdf5"
eval.out_path = "results/knn.hdf5"
eval.logger = get_logger("results/knn_entropy.log")

eval.quantity = "H"

eval.hyper_params = [1, 3, 5, 15, 50]
eval.sample_sizes = [100, 200, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
eval.seeds = range(1, 301)

# Create database (if not existing)
eval.create_database()
eval.create_group()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # UNIFORM # # # # #

experiment = "uniform"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

true_h = np.log(dist_params[0][1]) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # NORMAL # # # # #

experiment = "normal"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

true_h = 0.5 * np.log(2 * np.pi * (dist_params[0][1]**2)) + 0.5 # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # NORMAL-MIXTURE # # # # #

experiment = "normal-mixture"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

def pdf_normal(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.norm(loc=l, scale=s).pdf(x) * w
    return y

def h_normal(x, params):
    p = pdf_normal(x, params)
    return -1 * p * np.log(p)

norm_lims = [[-15, 25]]

true_h = nquad(h_normal, norm_lims, args=(dist_params,))[0] # Numerical Integration Result

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # EXPONENTIAL # # # # #

experiment = "exponential"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

true_h = 1 - np.log(1/dist_params[0][1]) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # BIVARIATE-NORMAL # # # # #

experiment = "bivariate-normal"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

d = len(dist_params[0][1])
true_h = 0.5 * np.log((2 * np.pi * np.exp(1)) ** d * np.linalg.det(dist_params[0][1])) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # BIVARIATE-NORMAL-MIXTURE # # # # #

experiment = "bivariate-normal-mixture"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

def pdf_mnorm(x, y, params):
    z = 0.0
    for dist in params:
        l, s, w = dist
        z += stats.multivariate_normal(mean=l, cov=s).pdf(np.dstack((x, y))) * w
    return z

def h_mnorm(x, y, params1):
    p = pdf_mnorm(x, y, params1)
    return -1 * p * np.log(p)

binorm_lims = [[-7, 7], [-7, 7]]

true_h = nquad(h_mnorm, binorm_lims, args=(dist_params,))[0] # Numerical Integration Result

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # GAMMA-EXPONENTIAL # # # # #

experiment = "gexp"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

tetha = dist_params[0][0]
true_h = 1 +  tetha - tetha * digamma(tetha) + np.log(gamma(tetha)) - np.log(1.0) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # 4D-GAUSSIAN # # # # #

experiment = "4d-gaussian"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

d = len(dist_params[0][1])
true_h = 0.5 * np.log((2 * np.pi * np.exp(1)) ** d * np.linalg.det(dist_params[0][1])) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")

# # # # # 10D-GAUSSIAN # # # # #

experiment = "10d-gaussian"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
   dist_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])

d = len(dist_params[0][1])
true_h = 0.5 * np.log((2 * np.pi * np.exp(1)) ** d * np.linalg.det(dist_params[0][1])) # Reference

start_time = time.perf_counter()
eval.evaluate(experiment, calc_knn_entropy, 1)
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_single_to_hdf5(experiment, true_h)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True entropy: {true_h:.3f} nats")