## Original Code
The original code can be found in [here](https://github.com/amazon-science/earth-forecasting-transformer).

For a detailed explanation of the methods, please refer to the paper: 
[Earthformer: Exploring Space-Time Transformers for Earth System Forecasting](https://arxiv.org/pdf/2207.05833)


## Usage
Commands for training and testing the model 

```bash
#axial
python tune_cuboid_solarcube.py --cfg cfg_solarcube_tf_axial.yaml

#divided space time
python tune_cuboid_solarcube.py --cfg cfg_solarcube_tf_dividest.yaml

#video swin
python tune_cuboid_solarcube.py --cfg cfg_solarcube_tf_swin.yaml
```

