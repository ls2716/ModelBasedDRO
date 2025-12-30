## DRO for insurance pricing problem

This directory contains the code to reproduce the experiments for the insurance pricing problem.

The main folders of this directory are:
- data: contains the data files used in the experiments.
- scripts: contains utility scripts used in the experiments.
- four experiment folders:
    - DRO: contains code for KL-DRO for the insurance pricing problem
    - DRO_chi: contains code for Chi-squared DRO for the insurance pricing problem
    - DRO_oos: contains code for out-of-sample evaluation of the KL-DRO
    - DRO_oos_chi: contains code for out-of-sample evaluation of the Chi-squared DRO

To first run the experiments, make sure you have the atoti data.csv in the /data/atoti/data.csv path.

For each of the experiment folders, the workflow is the same:
1. Enter the experiment folder (e.g., `cd DRO`)
2. Run the optimisation script:
    ```bash
    python experiment_optimise.py
    ```
3. Run the evaluation script:
    ```bash
    python experiment_evaluate.py
    ```
4. Run the plotting script to generate the plots:
    ```bash
    python experiment_plot.py
    ```

Each of this script is defined by hyperparameters set in the __main__ section of the script. You can modify these hyperparameters to change the behaviour of the experiments. The evaluaation and plotting scripts have to have consistent hyperparameters with the optimisation script to correctly load the results.