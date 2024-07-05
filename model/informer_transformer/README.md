## Original Code
The original code can be found in [here](https://github.com/zhouhaoyi/Informer2020).

For a detailed explanation of the methods, please refer to the paper: 
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436). 


## Usage
Commands for training and testing the model 

```bash
# short term informer
python -u main_informer.py --model informer_noT --data_path solarsat_point_if_ --seq_len 12 --label_len 12 --pred_len 12

# short term transformer
python -u main_transformer.py --model transformer_noT --data_path solarsat_point_if_long_ --seq_len 96 --label_len 96 --pred_len 96

# long term informer
python -u main_informer.py --model informer --data_path solarsat_point_if_ --seq_len 12 --label_len 12 --pred_len 12

# long term transformer
python -u main_transformer.py --model transformer_noT --data_path solarsat_point_if_long_ --seq_len 96 --label_len 96 --pred_len 96
```

More parameter information please refer to `main_informer.py` and `main_transformer.py`.

