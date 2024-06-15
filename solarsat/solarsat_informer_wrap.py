import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import codecs
import h5py
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader
from typing import Optional
import pytorch_lightning as pl
import sys
from .solarsat_dataloader import SOLARSATDataloader
from solarsat.timefeatures import time_features
from solarsat.utils import change_layout_np, generate_time_all, genertate_time

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
TYPES    = ['vis047','vis086','ir133','dsr','sza','insitu']
DEFAULT_DATA_HOME = os.path.abspath(os.path.join( '..', '..', 'data', 'geonex_sat'))
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarsat_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarsat_insitu.csv'

class SOLARSAT(data.Dataset):

    def __init__(self,
                 label_len=8,
                 data_path='solarsat_image_train.npz',
                 load_from_disk=True,
                 tile_list=[1],
                 year_list=['2018'],
                 x_img_types=['ssr'],
                 y_img_types=['ssr'],
                 input_len = 8,
                 output_len = 12,
                 sample_interval = 4,
                 downscale_s = 20,
                 downscale_t = False,
                 tile_all=DEFAULT_TILELIST,
                 start_date=None,
                 end_date=None,
                 datetime_filter=None,
                 catalog_filter=None,
                 unwrap_time=False,
                 output_type=np.float32,
                 normalize_x=None,
                 normalize_y=None,
                 normalize_max=False,
                 point_based=False,
                 layout = 'NTCHW',
                 train=True,
                 batch_size=1
                 ):
        super(SOLARSAT, self).__init__()
        self.load_from_disk = load_from_disk
        self.layout = layout
        self.input_len = input_len
        self.output_len = output_len
        self.label_len = label_len
        if self.load_from_disk:
            data = np.load(data_path,allow_pickle=True)
            self.data = data['arr_0'][:,0,:,:]
            self.labels = data['arr_1'][:,0,:,:]
            self.data = np.transpose(self.data, (0, 2, 1))
            self.labels = np.transpose(self.labels, (0, 2, 1))
            self.data = np.nan_to_num(self.data, nan=0)
            self.labels = np.nan_to_num(self.labels, nan=0)
            self.df_index = pd.DataFrame(data['arr_2'], columns=['tile', 'year', 'start_index'])
            self.df_index['start_time']=self.df_index.apply(lambda x: genertate_time(tile_id=x.tile,start=x.start_index, year=x.year, interval=15), axis=1)
            print('data shape', self.data.shape)
            # self.data=change_layout_np(self.data, in_layout= 'NCTHW', out_layout=self.layout)
            # self.labels=change_layout_np(self.labels, in_layout= 'NCTHW', out_layout=self.layout)
        
        self.solarsat_dataloader = SOLARSATDataloader(
                tile_list=tile_list,
                year_list=year_list,
                x_img_types=x_img_types,
                y_img_types=y_img_types,
                input_len = input_len,
                output_len = output_len,
                sample_interval = sample_interval,
                downscale_s = downscale_s,
                downscale_t = downscale_t,
                tile_all=tile_all,
                start_date=start_date,
                end_date=end_date,
                datetime_filter=datetime_filter,
                catalog_filter=catalog_filter,
                unwrap_time=unwrap_time,
                output_type=output_type,
                normalize_x=normalize_x,
                normalize_y=normalize_y,
                normalize_max=normalize_max,
                layout=layout,
                point_based=point_based,
                batch_size=batch_size
            )
            
    def __len__(self):
        """
        How many batches to generate per epoch
        """
        if self.load_from_disk:
            return len(self.data)
        else:
            return self.solarsat_dataloader.__len__()

    def __getitem__(self, idx):
        """
        Simple wrapper of get_batch that allowed the class to be used as a generator    
        """
        if self.load_from_disk:
            seq_x = self.data[idx]
            context_part = seq_x[-self.label_len:, :]
            # print(seq_x.shape)
            # print(context_part.shape)
            # print(self.labels[idx])
            seq_y = np.concatenate([context_part, self.labels[idx]], axis=0)
            # seq_y = np.concatenate([self.labels[idx], seq_x[-self.label_len:,-1:]], axis=0)
            # print(seq_y.shape)
            
            start_time = self.df_index.loc[idx, 'start_time'][1]
            df=pd.DataFrame()
            df['date'] = pd.date_range(start=start_time, periods=self.input_len+self.output_len, freq='15T')   #short 15T, long H
            data_stamp = time_features(df, timeenc=1, freq='T')       #short T, long H
            # print(data_stamp.shape)
            seq_x_mark = data_stamp[:self.input_len]
            seq_y_mark = data_stamp[(self.input_len-self.label_len):(self.input_len+self.output_len)]
            # print('hh')
            # print(seq_x_mark.shape, seq_y_mark.shape)
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            return self.solarsat_dataloader.get_batch(idx)
    
  

# class SolarSatLightningDataModule(pl.LightningDataModule):
#     #raw_folder = '/gpfs/data1/lianggp/lir/forcast/'
#     #processed_folder = 'data/'
#     def __init__(self, dataset_oc = None,
#                  val_ratio=0.1, seed=123, batch_size: int = 10):
#         """
#         Parameters
#         ----------
#         root
#         val_ratio
#         batch_size
#         rescale_input_shape
#             For the purpose of testing. Rescale the inputs
#         rescale_target_shape
#             For the purpose of testing. Rescale the targets
#         """
#         super().__init__()

#         self.val_ratio = val_ratio
#         self.seed = seed
#         self.data_path = dataset_oc['dataset_path']     #if load from disk
#         self.batch_size = dataset_oc['batch_size']
#         self.load_from_disk = dataset_oc['load_from_disk']

#         self.train_tile_list = dataset_oc['train_tile_list']
#         self.train_year_list = dataset_oc['train_year_list']
#         self.test_tile_list = dataset_oc['test_tile_list']
#         self.test_year_list = dataset_oc['test_year_list']
#         self.x_img_types = dataset_oc['x_img_types']
#         self.y_img_types = dataset_oc['y_img_types']
#         self.input_len = dataset_oc['input_len']
#         self.output_len = dataset_oc['output_len']
#         self.sample_interval = dataset_oc['sample_interval']
#         self.downscale_s = dataset_oc['downscale_s']
#         self.downscale_t = dataset_oc['downscale_t']
#         self.tile_all=pd.read_csv(dataset_oc['tile_all'],low_memory=False)

#         self.datetime_filter = dataset_oc['datetime_filter']
#         self.catalog_filter = dataset_oc['catalog_filter']
#         self.start_date = dataset_oc['start_date']
#         self.end_date = dataset_oc['end_date']
#         self.unwrap_time = dataset_oc['unwrap_time']
#         self.output_type = dataset_oc['output_type']
#         self.normalize_x = dataset_oc['normalize_x']
#         self.normalize_y = dataset_oc['normalize_y']
#         self.normalize_max = dataset_oc['normalize_max']
#         self.point_based = dataset_oc['point_based']                                
        
                 
#     def prepare_data(self):
#         data = SOLARSAT(sample_interval=self.sample_interval, x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_y=self.normalize_y,normalize_x=self.normalize_x, data_path=self.data_path+'test.npz', tile_list=self.train_tile_list,year_list=self.train_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
#         return data
        
#     def setup(self, stage: Optional[str] = None):
#         if stage == "fit" or stage is None:
#             self.lstm_train_val = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_y=self.normalize_y,normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'train.npz',tile_list=self.train_tile_list,year_list=self.train_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
#             all_indices = range(len(self.lstm_train_val))
#             train_indices, val_indices = train_test_split(all_indices, test_size=self.val_ratio, random_state=self.seed)
#             self.lstm_train = Subset(self.lstm_train_val, train_indices)
#             self.lstm_val = Subset(self.lstm_train_val, val_indices)

#         if stage == "test" or stage is None:
#             self.lstm_test = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_y=self.normalize_y,normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)

#         if stage == "predict" or stage is None:
#             self.lstm_predict = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_y=self.normalize_y,normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
            
#     # def trainval_dataloader(self):
#     #     return DataLoader(self.lstm_train_val, batch_size=self.batch_size, shuffle=False, num_workers=15)
            
#     def train_dataloader(self):
#         return DataLoader(self.lstm_train, batch_size=self.batch_size, shuffle=True, num_workers=15)

#     def val_dataloader(self):
#         return DataLoader(self.lstm_val, batch_size=self.batch_size, shuffle=False, num_workers=15)

#     def test_dataloader(self):
#         return DataLoader(self.lstm_test, batch_size=self.batch_size, shuffle=False, num_workers=15)

#     def predict_dataloader(self):
#         return DataLoader(self.lstm_predict, batch_size=self.batch_size, shuffle=False, num_workers=15)

#     @property
#     def num_train_samples(self):
#         return len(self.lstm_train)

#     @property
#     def num_val_samples(self):
#         return len(self.lstm_val)

#     @property
#     def num_test_samples(self):
#         return len(self.lstm_test)

#     @property
#     def num_predict_samples(self):
#         return len(self.lstm_predict)
    
    
class SolarSatDataModule():
    def __init__(self, dataset_oc = None,
                 val_ratio=0.1, seed=123, batch_size: int = 10):
        """
        Parameters
        ----------
        root
        val_ratio
        batch_size
        rescale_input_shape
            For the purpose of testing. Rescale the inputs
        rescale_target_shape
            For the purpose of testing. Rescale the targets
        """
        super().__init__()

        self.val_ratio = val_ratio
        self.seed = seed
        self.data_path = dataset_oc['dataset_path']     #if load from disk
        self.load_from_disk = dataset_oc['load_from_disk']

        self.batch_size = dataset_oc['batch_size']

        self.train_tile_list = dataset_oc['train_tile_list']
        self.train_year_list = dataset_oc['train_year_list']
        self.test_tile_list = dataset_oc['test_tile_list']
        self.test_year_list = dataset_oc['test_year_list']
        self.x_img_types = dataset_oc['x_img_types']
        self.y_img_types = dataset_oc['y_img_types']
        self.input_len = dataset_oc['input_len']
        self.output_len = dataset_oc['output_len']
        self.sample_interval = dataset_oc['sample_interval']
        self.downscale_s = dataset_oc['downscale_s']
        self.downscale_t = dataset_oc['downscale_t']
        self.tile_all=pd.read_csv(DEFAULT_TILELIST,low_memory=False)

        self.datetime_filter = dataset_oc['datetime_filter']
        self.catalog_filter = dataset_oc['catalog_filter']
        self.start_date = dataset_oc['start_date']
        self.end_date = dataset_oc['end_date']
        self.unwrap_time = dataset_oc['unwrap_time']
        self.output_type = dataset_oc['output_type']
        self.normalize_x = dataset_oc['normalize_x']
        self.normalize_y = dataset_oc['normalize_y']
        self.normalize_max = dataset_oc['normalize_max']
        self.point_based = dataset_oc['point_based']
        
                 
    def prepare_data(self):
        data = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz', tile_list=self.train_tile_list,year_list=self.train_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
        return data
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.lstm_train_val = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'train.npz',tile_list=self.train_tile_list,year_list=self.train_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
            all_indices = range(len(self.lstm_train_val))
            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_ratio, random_state=self.seed)
            self.lstm_train = Subset(self.lstm_train_val, train_indices)
            self.lstm_val = Subset(self.lstm_train_val, val_indices)

        if stage == "test" or stage is None:
            self.lstm_test = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)

        if stage == "predict" or stage is None:
            self.lstm_predict = SOLARSAT(sample_interval=self.sample_interval,x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale_s=self.downscale_s, downscale_t=self.downscale_t, train=True)
         
    def trainval_dataloader(self):
        return DataLoader(self.lstm_train_val, batch_size=self.batch_size, shuffle=False, num_workers=15)
       
    def train_dataloader(self):
        return DataLoader(self.lstm_train, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader( self.lstm_val, batch_size=self.batch_size,shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.lstm_test, batch_size=self.batch_size,shuffle=False, num_workers=15)

    def predict_dataloader(self):
        return DataLoader(self.lstm_predict, batch_size=self.batch_size,shuffle=False, num_workers=15)

    @property
    def num_train_samples(self):
        return len(self.lstm_train)

    @property
    def num_val_samples(self):
        return len(self.lstm_val)

    @property
    def num_test_samples(self):
        return len(self.lstm_test)

    @property
    def num_predict_samples(self):
        return len(self.lstm_predict)
    