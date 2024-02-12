import os
import numpy as np

# Script to read the log files of all experiments and find the
# minimum, maximum and average times required per iteration.

path = "../data_evaluation/results"

# Find all logs in directory
path_logs = []
for file in os.listdir(path):
    if file[-4:] == ".log":
        path_logs.append(path + "/" + file)

# Interate through each line of all log files
for file in path_logs:
    print("\n" + file)
    with open(file) as f:
        res = {}
        for line in f.readlines():
            if line.find("(") != -1:
                # Find the distribution and hyperparameter
                case = line[line.find("(") + 1 : line.find(")")].split(",")
                distribution = case[0].strip()
                hyper_param = case[1].strip()

                # Find the time
                index = line.find("Time: ")
                sub = line[index : index + 15].split(" ")
                time = float(sub[1].strip())

                # Add to dictionary
                key = f"{distribution} - {hyper_param}"
                if key in res.keys():
                    res[key].append(time)
                else:
                    res[key] = [time]

        print(
            "{:<40s}{:<10s}{:<10s}{:<10}{:<10}{}".format(
                "Case", "MEAN", "MIN", "MAX", "COUNT", "LATEX"
            )
        )
        for key, val in res.items():
            res_mean = np.mean(val)
            res_min = np.min(val)
            res_max = np.max(val)
            print(
                f"{key:<40s}{res_mean:<10.3f}{res_min:<10.3f}{res_max:<10.3f}{len(val):<10}{res_mean:.3f}_{{{res_min:.3f}}}^{{{res_max:.3f}}}"
            )
