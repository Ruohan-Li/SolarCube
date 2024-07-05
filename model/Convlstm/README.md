## Original Code
The original code can be found in [here](https://github.com/jhhuang96/ConvLSTM-PyTorch).

For a detailed explanation of the methods, please refer to the paper: 
[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)


## Usage
Commands for training and testing the model 

```bash
#train
python model/ConvLSTM-PyTorch/main.py --convlstm --cfg cfg_solarsat_cl_test4.yaml --directory convlstm

#Test
python model/ConvLSTM-PyTorch/main.py --convlstm --cfg cfg_solarsat_tf_test4.yaml --directory convlstm --test
```

