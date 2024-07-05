## Original Code
The original code can be found in [here](https://github.com/amazon-science/earth-forecasting-transformer).

For a detailed explanation of the methods, please refer to the paper: 
[Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/pdf/2207.05833)


## Usage
Commands for training and testing the model 

```bash
#train
python train_cuboid_solarsat.py --cfg cfg_solarsat.yaml

#Test
python train_cuboid_solarsat.py --cfg cfg_solarsat.yaml --test
```

