# Model-based Distributionally Robust Optimisation for Pricing

In this folder, there is a complete set of numerical experiments for the model-based
distributionally robust optimisation (MBDRO) for pricing problem.

## Purpose of the project and the numerical experiments

The purpose of the numerical experiments is to show the application of model-based and
bayesian approach to DRO for the problem of pricing.

Current DRO approaches are often model-free which comes with a few shortcomings:

- They are not able to incorporate prior knowledge about the distribution of the data.
- They require finite action space.

Due to inability to consider the prior knowledge and structure of the distribution, they
require significant amount of data to provide good estimates of the optimised
quantities. While these methods apply the distributionally robust optimisation and thus
should be robust to errounous data distributions this is not efficient. Furthermore, the
worst-case distributions considered by these methods do not abide by the structure of
the environment. There is no guarantee that the worst-case distribution of outcomes and
context is reallistic.

Due to the requirement of finiteness of the action space, the DRO methods often cannot
be applied to optimisation problems where the action space is continuous, as is the case
for the problem of pricing.

The model-based approach and the bayesian model-based approach aim at addressing these
issues at the cost of the requirement for the input model of the environment and the
subsequent assumption that the model is correct.

Thus, the purpose of the numerical experiments are the following (in loose chronological
order):

- For data generated using a synthetic environment with logistic acceptance probability:
  - Show that model-based approach can work when the model-free cannot. I.e. Show that
    even for finite action space, the model-free approach fails to properly estimate the
    expectations while model-based approach can.
  - Show that bayesian model-based approach works better than just model-based as it
    considers the structure of the environment for worst-case distribution.
  - Show that the bayesian model-based approach works only for specific model parameters.
- For the real data:
    - Show that the logistic approximation works well for the real data.
    - Show that the parameters correspond to the case where there is little advatange for DRO.

