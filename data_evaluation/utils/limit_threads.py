import os

os.environ["OMP_NUM_THREADS"] = "1"
num_threads = os.environ["OMP_NUM_THREADS"]


def print_threads():
    print(f"Using {num_threads} thread(s).")
