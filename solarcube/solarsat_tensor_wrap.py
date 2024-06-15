from PIL import Image
import os
import os.path
import errno
import numpy as np
import pandas as pd
import codecs
import h5py
from skimage.measure import block_reduce
from sklearn.model_selection import train_test_split
from typing import Optional
import sys
print(sys.path)
from solarsat.solarsat_dataloader import SOLARSATDataloader
from solarsat.solarsat_dataloader import save_all_data
from solarsat.utils import change_layout_np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
TYPES    = ['vis047','vis086','ir133','dsr','sza','insitu']
DEFAULT_DATA_HOME = os.path.abspath(os.path.join( '..', '..', 'data', 'geonex_sat'))
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarsat_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarsat_insitu.csv'

class SOLARSAT():

    def __init__(self,
                 data_path='solarsat_image_train.npz',
                 load_from_disk=True,
                 tile_list=[1],
                 year_list=['2018'],
                 x_img_types=['ssr'],
                 y_img_types=['ssr'],
                 input_len = 8,
                 output_len = 12,
                 sample_interval = 4,
                 downscale = 20,
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
        if self.load_from_disk:
            data = np.load(data_path)
            self.data = data['arr_0']
            self.labels = data['arr_1']
            print(self.data.shape)
            self.data=change_layout_np(self.data, in_layout= 'NCTHW', out_layout=self.layout)
            self.labels=change_layout_np(self.labels, in_layout= 'NCTHW', out_layout=self.layout)
        
        self.solarsat_dataloader = SOLARSATDataloader(
                tile_list=tile_list,
                year_list=year_list,
                x_img_types=x_img_types,
                y_img_types=y_img_types,
                input_len = input_len,
                output_len = output_len,
                sample_interval = sample_interval,
                downscale = downscale,
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
            return self.data[idx], self.labels[idx]
        else:
            return self.solarsat_dataloader.get_batch(idx)
    
  
    
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
        self.downscale = dataset_oc['downscale']
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
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.lstm_train_val = SOLARSAT(x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x, load_from_disk=self.load_from_disk,data_path=self.data_path+'train.npz',tile_list=self.train_tile_list,year_list=self.train_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale=self.downscale, train=True)
        
        if stage == "test" or stage is None:
            self.lstm_test = SOLARSAT(x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale=self.downscale, train=True)

        if stage == "predict" or stage is None:
            self.lstm_predict = SOLARSAT(x_img_types=self.x_img_types, y_img_types=self.y_img_types, point_based=self.point_based, normalize_x=self.normalize_x,load_from_disk=self.load_from_disk,data_path=self.data_path+'test.npz',tile_list=self.test_tile_list,year_list=self.test_year_list, input_len=self.input_len, output_len=self.output_len, normalize_max=self.normalize_max, downscale=self.downscale, train=True)

    @property
    def num_train_samples(self):
        return len(self.lstm_train_val)

    @property
    def num_val_samples(self):
        return len(self.lstm_val)

    @property
    def num_test_samples(self):
        return len(self.lstm_test)

    @property
    def num_predict_samples(self):
        return len(self.lstm_predict)
    
   
   

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None, type=str)
    args = parser.parse_args()
    oc_from_file = OmegaConf.load(open(args.cfg, "r"))
    dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
    random_seed = oc_from_file.optim.seed
    name_prefix=str(dataset_oc['train_tile_list'])+'_'+str(dataset_oc['test_tile_list'])+'_'+str(dataset_oc['x_img_types'])+str(dataset_oc['y_img_types'])
    print(name_prefix)
            
    dm = SolarSatDataModule(dataset_oc=dataset_oc)
    dm.setup()
    
    print('saving test data...')
    x_test=[]
    y_test=[]
    x_test,y_test = save_all_data(dm.lstm_test.solarsat_dataloader)
    x_test_np = np.array(x_test)
    y_test_np = np.array(y_test)
    print(x_test_np.shape, y_test_np.shape)
    np.savez_compressed('solarsat_point2_test.npz', x_test_np, y_test_np, dm.lstm_test.solarsat_dataloader._samples)

    print('saving train data...')
    x_train=[]
    y_train=[]
    x_train,y_train = save_all_data(dm.lstm_train_val.solarsat_dataloader)
    x_train_np = np.array(x_train)
    y_train_np = np.array(y_train)
    print(x_train_np.shape, y_train_np.shape)
    np.savez_compressed('solarsat_point2_train.npz', x_train_np, y_train_np,dm.lstm_train_val.solarsat_dataloader._samples)
    # t = tqdm(trainvalLoader, leave=False, total=len(trainvalLoader))
    # for i, ( inputVar,targetVar) in enumerate(t):
    #     x_train.append(inputVar.detach().cpu())  # Ensure tensors are detached and moved to CPU
    #     y_train.append(targetVar.detach().cpu())
    # x_train_tensor = torch.cat(x_train, dim=0)  # Adjust 'dim' as necessary for your data
    # y_train_tensor = torch.cat(y_train, dim=0)
    # x_train_np = x_train_tensor.numpy()
    # y_train_np = y_train_tensor.numpy()
    # print(x_train_np.shape, y_train_np.shape)
    # np.savez_compressed('solarsat_image2_train.npz', x_train_np, y_train_np)