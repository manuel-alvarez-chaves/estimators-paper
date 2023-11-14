# File/OS management
import os, sys, time
import h5py, ast

# Required modules
import numpy as np
from scipy import stats
from scipy.special import gamma, digamma
from scipy.integrate import nquad

from utils.tools import get_logger
from utils.bin_evaluators import Evaluator_BIN

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

os.chdir(sys.path[0]) # Set location of file to CWD

# Evaluator attributes
eval = Evaluator_BIN()
eval.data_path = "data/data.hdf5"
eval.out_path = "results/bin.hdf5"
eval.logger = get_logger("results/bin_kld.log")

eval.quantity = "KLD"

eval.hyper_params = ["scott", "fd", "sturges"]
eval.sample_sizes = [100, 200, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
eval.seeds = range(1, 501)

# Create database (if not existing)
eval.create_database()
eval.create_group()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # UNIFORM # # # # #

experiment = "uniform"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

true_kld = np.log(dist2_params[0][1] / dist1_params[0][1])

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "uniform||uniform", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # NORMAL # # # # #

experiment = "normal"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

true_kld = 0.5 * (
    (dist1_params[0][1]/dist2_params[0][1]) ** 2 +
    (dist2_params[0][0] - dist1_params[0][0]) ** 2 / (dist2_params[0][1] ** 2) -
    1 + np.log((dist2_params[0][1]**2)/(dist1_params[0][1]**2))
) # Reference

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "normal||normal", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # NORMAL-MIXTURE # # # # #

experiment = "normal-mixture"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

def pdf_normal(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.norm(loc=l, scale=s).pdf(x) * w
    return y

def kld_normals(x, params1, params2):
    p = pdf_normal(x, params1)
    q = pdf_normal(x, params2)
    return p * np.log(p / q)

norm_lims = [[-15, 25]]

true_kld = nquad(kld_normals, norm_lims, args=(dist1_params, dist2_params,))[0] # Numerical Integration Solution


start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "normal-mixture||normal", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # EXPONENTIAL # # # # #

experiment = "exponential"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

true_kld = np.log(1 / dist1_params[0][1]) - np.log(1 / dist2_params[0][1]) + dist1_params[0][1] / dist2_params[0][1] - 1 # Reference

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "exp||exp", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # BIVARIATE NORMAL # # # # #

experiment = "bivariate-normal"
# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

m1, s1, _ = dist1_params[0]
m2, s2, _ = dist2_params[0]
m1, s1, m2, s2 = [np.array(p) for p in [m1, s1, m2, s2]]

true_kld = 0.5 * (
    np.log(np.linalg.det(s2)/np.linalg.det(s1)) + 
    np.trace(np.linalg.inv(s2) @ s1) +
    (m2 - m1).T @ np.linalg.inv(s2) @ (m2 - m1) -
    len(m2)
)

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "bivariate-normal||bivariate-normal", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # BIVARIATE-NORMAL-MIXTURE # # # # #

experiment = "bivariate-normal-mixture"
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

def pdf_mnorm(x, y, params):
    z = 0.0
    for dist in params:
        l, s, w = dist
        z += stats.multivariate_normal(mean=l, cov=s).pdf(np.dstack((x, y))) * w
    return z

def kld_mnorms(x, y, params1, params2):
    p = pdf_mnorm(x, y, params1)
    q = pdf_mnorm(x, y, params2)
    return p * np.log(p / q)

mnorm_lims = [[-7, 7], [-7, 7]]

true_kld = nquad(kld_mnorms, mnorm_lims, args=(dist1_params, dist2_params,))[0]

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "bivariate-normal-mixture||bivariate-normal", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # GAMMA-EXPONENTIAL # # # # #

experiment = "gexp"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

def pdf_gamma_exponential(x, y, params):
    z = 0.0
    for dist in params:
        t, w = dist
        z += (1 / gamma(t)) * (x**t) * np.exp(-x - x * y) * w
    return z

def kld_gamma_exponentials(x, y, params1, params2):
    p = pdf_gamma_exponential(x, y, params1)
    q = pdf_gamma_exponential(x, y, params2)
    return p * np.log(p / q)

gexp_lims = [[0, 15], [0, 12]]

true_kld = nquad(kld_gamma_exponentials, gexp_lims, args=(dist1_params, dist2_params,))[0]

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "gexp||gexp", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")

# # # # # 4D-GAUSSIAN # # # # #

experiment = "4d-gaussian"

# Calculate Truth
with h5py.File(eval.data_path, "r") as f:
    dist1_params = ast.literal_eval(f[experiment]["p"].attrs["hyper_params"])
    dist2_params = ast.literal_eval(f[experiment]["q"].attrs["hyper_params"])

def kld_scipy_mnorm(d1, d2):
    a = np.log(np.linalg.det(d2.cov) / np.linalg.det(d1.cov))
    b = np.trace(np.linalg.inv(d2.cov) @ d1.cov)
    c = (d1.mean - d2.mean) @ np.linalg.inv(d2.cov) @ (d1.mean - d2.mean).T
    n = len(d1.mean)

    kld = 0.5 * (a + b) + 0.5 * (c - n)
    return kld

dist1 = stats.multivariate_normal(mean=dist1_params[0][0], cov=dist1_params[0][1])
dist2 = stats.multivariate_normal(mean=dist2_params[0][0], cov=dist2_params[0][1])

true_kld = kld_scipy_mnorm(dist1, dist2)

start_time = time.perf_counter()
eval.evaluate_kld(experiment, "scott")
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))

# Save
eval.write_double_to_hdf5(experiment, "4dgauss||4dgauss", true_kld)
eval.logger.info(f"FINISHED {experiment.upper()} - Elapsed time: {elapsed_time} - True KLD: {true_kld:.3f} nats")