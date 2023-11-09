import numpy as np
from scipy import stats
from scipy.special import gamma

import time
from utils import get_logger, create_database
from utils import get_samples, save_to_file

logger = get_logger("data_generation.log")
fpath = "data.hdf5"

create_database(fpath)

seeds = range(1, 501)
sample_sizes = [100, 200, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000]

##### UNIFORM DISTRIBUTION #####

case = "uniform"

def pdf_uniform(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.uniform(loc=l, scale=s).pdf(x) * w
    return y

unif1_params = [[0.5, 1.5, 1.0]]
unif2_params = [[0.0, 2.0, 1.0]]
unif_lims = [[-0.1, 2.5]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_uniform,
                              limits = unif_lims,
                              n_samples = n,
                              seed = s,
                              params = unif1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", unif1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_uniform,
                              limits = unif_lims,
                              n_samples = n,
                              seed = s,
                              params = unif2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", unif2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### NORMAL DISTRIBUTION #####

case = "normal"

def pdf_normal(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.norm(loc=l, scale=s).pdf(x) * w
    return y

norm1_params =  [[-2.5, 2.5, 1.0]]
norm2_params =  [[0.0, 3.15, 1.0]]
norm_lims = [[-15, 25]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_normal,
                              limits = norm_lims,
                              n_samples = n,
                              seed = s,
                              params = norm1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", norm1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_normal,
                              limits = norm_lims,
                              n_samples = n,
                              seed = s,
                              params = norm2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", norm2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### NORMAL MIXTURE DISTRIBUTION #####

case = "normal-mixture"

norm1_params =  [[-2.5, 2.5, 0.5], [2.5, 1.0, 0.5]]
norm2_params =  [[0.0, 3.15, 1.0]]
norm_lims = [[-15, 25]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_normal,
                              limits = norm_lims,
                              n_samples = n,
                              seed = s,
                              params = norm1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", norm1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_normal,
                              limits = norm_lims,
                              n_samples = n,
                              seed = s,
                              params = norm2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", norm2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### EXPONENTIAL DISTRIBUTION #####

case = "exponential"

def pdf_exponential(x, params):
    y = 0.0
    for dist in params:
        l, s, w = dist
        y += stats.expon(scale=s).pdf(x) * w
    return y

expon1_params =  [[0.0, 0.5, 1.0]]
expon2_params =  [[0.0, 2.0, 1.0]]
expon_lims = [[0, 20]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_exponential,
                              limits = expon_lims,
                              n_samples = n,
                              seed = s,
                              params = expon1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", expon1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_exponential,
                              limits = expon_lims,
                              n_samples = n,
                              seed = s,
                              params = expon2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", expon2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### BIVARIATE NORMAL DISTRIBUTION #####

case = "bivariate-normal"

def pdf_mnorm(x, y, params):
    z = 0.0
    for dist in params:
        l, s, w = dist
        z += stats.multivariate_normal(mean=l, cov=s).pdf(np.dstack((x, y))) * w
    return z

mnorm1_params = [[[-2, 0], [[1, -0.5], [-0.5, 1]], 1.0]]
mnorm2_params = [[[0, 0], [[5, 0], [0, 1]], 1.0]]
mnorm_lims = [[-10, 10], [-10, 10]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_mnorm,
                              limits = mnorm_lims,
                              n_samples = n,
                              seed = s,
                              params = mnorm1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", mnorm1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_mnorm,
                              limits = mnorm_lims,
                              n_samples = n,
                              seed = s,
                              params = mnorm2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", mnorm2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### BIVARIATE NORMAL MIXTURE DISTRIBUTION #####

case = "bivariate-normal-mixture"

mnorm1_params = [
    [[-2, 0], [[1, -0.5], [-0.5, 1]], 0.5],
    [[2, 0], [[1, 0.5], [0.5, 1]], 0.5]
]
mnorm2_params = [[[0, 0], [[5, 0], [0, 1]], 1.0]]
mnorm_lims = [[-10, 10], [-10, 10]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_mnorm,
                              limits = mnorm_lims,
                              n_samples = n,
                              seed = s,
                              params = mnorm1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", mnorm1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_mnorm,
                              limits = mnorm_lims,
                              n_samples = n,
                              seed = s,
                              params = mnorm2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", mnorm2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### GAMMA EXPONENTIAL DISTRIBUTION #####

case = "gexp"

def pdf_gamma_exponential(x, y, params):
    z = 0.0
    for dist in params:
        t, w = dist
        z += (1 / gamma(t)) * (x**t) * np.exp(-x - x * y) * w
    return z

gexp1_params = [[3, 1]]
gexp2_params = [[4, 1]]
gexp_lims = [[0, 15], [0, 12]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_gamma_exponential,
                              limits = gexp_lims,
                              n_samples = n,
                              seed = s,
                              params = gexp1_params)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", gexp1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = get_samples(func = pdf_gamma_exponential,
                              limits = gexp_lims,
                              n_samples = n,
                              seed = s,
                              params = gexp2_params)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", gexp2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### 4D GAUSSIAN DISTRIBUTION #####

case = "4d-gaussian"

mu1 = [0.1, 0.3, 0.6, 0.9]
sigma1 = [
    [1, 0.5, 0.5, 0.5],
    [0.5, 1, 0.5, 0.5],
    [0.5, 0.5, 1, 0.5],
    [0.5, 0.5, 0.5, 1]
]
dist1 = stats.multivariate_normal(mean=mu1, cov=sigma1)
gauss4d_1_params = [[mu1, sigma1, 1.0]]

mu2 = [0, 0, 0, 0]
sigma2 = [
    [1, 0.1, 0.1, 0.1],
    [0.1, 1, 0.1, 0.1],
    [0.1, 0.1, 1, 0.1],
    [0.1, 0.1, 0.1, 1]
]
dist2 = stats.multivariate_normal(mean=mu2, cov=sigma2)
gauss4d_2_params = [[mu2, sigma2, 1.0]]


logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = dist1.rvs(size=n, random_state=s)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", gauss4d_1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = dist2.rvs(size=n, random_state=s)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", gauss4d_2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")

##### 10D GAUSSIAN DISTRIBUTION #####

case = "10d-gaussian"
d = 10

mu1 = np.zeros(shape=(d))
sigma1 = 0.9 * np.ones(shape=(d, d))
np.fill_diagonal(sigma1, 1.0)
dist1 = stats.multivariate_normal(mean=mu1, cov=sigma1)
gauss10d_1_params = [[mu1, sigma1, 1.0]]

mu2 = np.zeros(shape=(d))
sigma2 = 0.1 * np.ones(shape=(d, d))
np.fill_diagonal(sigma2, 1.0)
dist2 = stats.multivariate_normal(mean=mu2, cov=sigma2)
gauss10d_2_params = [[mu2, sigma2, 1.0]]

logger.info(f"Started case: {case.upper()}")
start = time.time()

data1 = {}
for s in seeds:
    data1[s] = {}
    for n in sample_sizes:
        samples = dist1.rvs(size=n, random_state=s)
        data1[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "p", gauss10d_1_params, sample_sizes, seeds, data1)

data2 = {}
for s in seeds:
    data2[s] = {}
    for n in sample_sizes:
        samples = dist2.rvs(size=n, random_state=s)
        data2[s][n] = samples
    if s % 100 == 0: logger.info(f"Seed: {s}")
save_to_file(fpath, case, "q", gauss10d_2_params, sample_sizes, seeds, data2)

elapsed_time = time.time() - start
logger.info(f"FINISHED - Elapsed time: {elapsed_time:.2f} s")