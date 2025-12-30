## DRO for newsvendor problem

This folder contains code to reproduce DRO experiments for the newsvendor problem.

### Running the code - Delta sensitivity analysis

To run the delta sensitivity analysis, first run the script which conducts the experiments:

```bash
python experiment_loop_delta.py
```

This script will solve the DRO newsvendor problem for different values of delta (both evaluation and training deltas) and store the results in the `results/experiment_results.json` file.

To visualize the results, run the plotting script:

```bash
python plot_loop_delta.py
```

This will generate plots showing the performance of the DRO newsvendor model for different delta values. The plots will be saved in the `plots/` directory.


### Running the code - Out-of-sample performance analysis


To run the out-of-sample performance analysis, first execute the main experiment script:

```bash
python experiment_oos.py
```

The script runs the out-of-sample performance experiments parametrised by two parameters: no_samples and delta. The results will be saved in the `results/results_delta_{delta}_no_samples_{no_samples}.json` files.

To visualize the out-of-sample performance results, run the plotting scripts:

For analysis of a single delta and no_samples pair, use: (set parameters in the script)
```bash
python plot_oos.py
```
This script will generate plots for the specified delta and no_samples values, saved in the `plots/` directory.

For analysis across multiple deltas and no_samples values, use:
```bash
python plot_oos_loop.py
```
or
```bash
python plot_oos_loop_no_var.py
```
These scripts will generate plots that illustrate the out-of-sample performance of the DRO newsvendor model under various configurations. The plots will be saved in the `plots/` directory. The first script accounts for both between-experiement and within-experiment variance, while the second script focuses solely on between-experiment variance.

