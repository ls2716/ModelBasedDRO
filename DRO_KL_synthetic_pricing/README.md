## KL-DRO for Synthetic Pricing

This directory contains code to reproduce the synthetic pricing experiments using KL-DRO (Kullback-Leibler Divergence Distributionally Robust Optimization) as described in our research.

### Running the Experiments

To run the experiments, execute the following command in your terminal:

```bash
python experiment_dro.py
```

This will run the KL-DRO algorithm on the synthetic pricing dataset and output the results.

The script `experiment_dro.py` includes all necessary components to set up the KL-DRO framework, including data generation, model training, and evaluation. The results depend on three hyperparameters:
- `alpha` - the slope of the logistic demand function
- `beta` - the intercept of the logistic demand function
- random seed - for reproducibility of results.

