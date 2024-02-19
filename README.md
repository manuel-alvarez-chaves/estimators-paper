This is a repository for our paper on: "Evaluating Density- and Nearest Neighbor-based Methods to Accurately Estimate Information-Theoretic Quantities from Multi-Dimensional Sample Data".

The projects is organizes as follows:

├── analysis_results\
│   ├── plots\
├── data_evaluation\
│   ├── data\
│   ├── notebooks\
│   ├── results\
│   ├── utils\
│   ├── (...) scripts
├── data_generation\
├── README.md
└── .gitignore

Initially data is generated and stored in the data_evaluation/data directory using the script in the data_generation/ directory. The data for the experiments is stored as a HDF5 databse. The scripts at the root of the directory data_evaluation/ are used to read the data and perform the experiments. Results are stored at the results/ directory. The notebooks/ directory serves as an archive for as the development of the workflow to test each estimator and the contents of each notebook are generally the same as the code in the scripts. Log files describe the history of the project.

All results produced using the [UNITE Toolbox](https://github.com/manuel-alvarez-chaves/unite_toolbox).
Contact: [manuel.alvarez-chaves@simtech.uni-stuttgart.de](manuel.alvarez-chaves@simtech.uni-stuttgart.de)
