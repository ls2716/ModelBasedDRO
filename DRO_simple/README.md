## Simple DRO example

This folder contains a simple experiment which shows when DRO (in this case KL-DRO) can provide a robust policy.

### Running the experiment

To run the experiment, execute the following command in your terminal:

```bash
python simple_KL.py
```

This will run a simple KL-DRO experiment which will result in a single plot ('simple_KL_robust_profit.png') being generated and saved to the current directory. The plot will show the performance of the DRO policy compared to a standard policy under different environment shift.