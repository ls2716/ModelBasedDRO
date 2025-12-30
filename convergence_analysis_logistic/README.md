## Converge analysis for DRO for logistic regression problem

This folder contains experiments that analyze the convergence behavior of DRO objective (chi-squared DRO) for logistic regression problem.


### Running the experiment

The repository provides three scripts to run the convergence analysis experiments:

#### sparse_data_experiment.py

To run the sparse data experiment, execute the following command in your terminal:

```bash
python sparse_data_experiment.py
```

The script runs a single experiment showing convergence of the DRO objective for sparse data. It will generate and save a plot ('mf_vs_mb_example.png') which will illustrate the objective values for both parametric and pure data-driven DRO evaluation.

#### sparse_data_loop.py

To run the sparse data loop experiment, execute the following command in your terminal:

```bash
python sparse_data_loop.py
```

This script runs multiple experiments based on the configuration i.e. env parameters specified in the main part of the script. It will run the experiments and save them in the 'results' folder as .pkl files. The input parameter that can be modified is the beta parameter of the logistic regression.

#### plot_errors.py

To plot the errors from the experiments, execute the following command in your terminal:

```bash
python plot_errors.py
```

The script will generate comparison plots in the 'plots' folder based on the results saved in the 'results' folder. You can modify the input parameters in the main part of the script to specify which experiments to plot.