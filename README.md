# Model-Based DRO

This repository contains the code accompanying the paper:  
“Parametric Phi-Divergence-Based Distributionally Robust Optimization for Insurance Pricing”  
published in the *Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF 2025)*.

The code also supports the research presented in Chapter 2 of the PhD thesis:  
“Algorithmic Pricing in Multi-agent, Competitive Markets”  
by Lukasz Sliwinski, 2025, University of Edinburgh.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install in editable mode:

```bash
pip install -e .
```

## Usage

Each part of the numerical experiments is split into different folders. As such:
- 'DRO_insurance_pricing' contains the code for the insurance pricing experiments, which include:
    - data in the 'data' folder,
    - scripts in the 'scripts' folder which define data tranformation and utility functions,
    - DRO optimisation for the insurance pricing problem using KL divergence in the 'DRO' folder,
    - DRO optimisation for the insurance pricing problem using Chi-squared divergence in the 'DRO_chi' folder,
    - out-of-samples experiments for the KL-DRO in the 'DRO_oos' folder,
    - out-of-samples experiments for the Chi-squared DRO in the 'DRO_oos_chi' folder,
- 'DRO_KL_synthetic_pricing' contains the code for the synthetic pricing experiments using KL divergence,
- 'DRO_chi_synthetic_pricing' contains the code for the synthetic pricing experiments using Chi-squared divergence,
- 'DRO_simple' which contains the simple example where KL-DRO works
- 'DRO_newsvendor' which contains application of KL-DRO to the newsvendor problem,
- 'convergence_analysis_logistic' which contains the convergence analysis of DRO for logistic regression.
- 'convergence_analysis_voting' which contains the convergence analysis of DRO for voting problem i.e., classification of voters based on their demographics.

Each folder contains a README.md file with instructions on how to run the experiments and plot the results.

