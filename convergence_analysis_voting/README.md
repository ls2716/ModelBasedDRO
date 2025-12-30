## Convergence analysis voting

This directory contains scripts and data related to the convergence analysis DRO optimisation (chi-squared DRO) applied to a voting problem i.e. logistic classification mimicing the voting dataset.

### Running the experiment

There is one script which runs the convergence analysis experiment:

```bash
python full_analysis_DRO.py
```

This runs the experiment based on the random seed and the data generation seed set in the script. The script will generate a number of plots and save them to the current directory. The results are saved in the 'results_dro' folder in 'seed_{random_seed}_{data_generation_seed}_{delta}' subfolder as policy and experiments results in 'results.json' file.


#### Individual plots

The individual plots can be generated using 'plot_error.py' script with appriopriate seed in the __main__ section of the script:

```bash
python plot_seed.py 
```

This will generate plots showing the convergence of the chi-squared DRO evaluation error as more samples are fed into the evaluation. The plots will be saved to the results folder corresponding to the input seed (random and data generation seed).
```

#### Summary plots

The summary plots can be generated using 'plot_error.py' script with appriopriate seeds in the __main__ section of the script:

```bash
python plot_error.py 
```

This will create error convergence plots averaged over multiple random seeds for given data generation seed and delta value. The plots will be saved to the 'results_dro' folder.


