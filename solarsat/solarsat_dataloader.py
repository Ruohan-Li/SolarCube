import os
import numpy as np
import pandas as pd
import h5py
from skimage.measure import block_reduce
from solarsat.utils import change_layout_np
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
TYPES    = ['vis047','vis086','ir133','ssr','sza','insitu']
# _curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_DATA_HOME = os.path.abspath(os.path.join( '..', '..', 'data', 'geonex_sat'))
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarsat_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarsat_insitu.csv'

SCALE_SOLARSAT = {'vis047': 0.39,  # Not utilized in original paper
                          'vis086': 0.32,
                          'ir133': 17.13,
                          'ssr': 279.11,
                          'sza': 20.19,
                          'insitu' : 279.49}
OFFSET_SOLARSAT = {'vis047': 0.56,  # Not utilized in original paper
                          'vis086': 0.55,
                          'ir133': 258.55,
                          'ssr': 186.96,
                          'sza': 56.67,
                          'insitu' : 193.44}

class SOLARSATDataloader:
    """
    Sequence class for generating batches from SEVIR

    Parameters
    ----------
    catalog  str or pd.DataFrame
        name of SEVIR catalog file to be read in, or an already read in and processed catalog
    x_img_types  list
        List of image types to be used as model inputs.  For types, run SEVIRSequence.get_types()
    y_img_types  list or None
       List of image types to be used as model targets (if None, __getitem__ returns only x_img_types )
    sevir_data_home  str
       Directory path to SEVIR data
    catalog  str
       Name of SEVIR catalog CSV file.
    start_date   datetime
       Start time of SEVIR samples to generate
    end_date    datetime
       End time of SEVIR samples to generate
    datetime_filter   function
       Mask function applied to time_utc column of catalog (return true to keep the row).
       Pass function of the form   lambda t : COND(t)
       Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
    catalog_filter  function
       Mask function applied to entire catalog dataframe (return true to keep row).
       Pass function of the form lambda catalog:  COND(catalog)
       Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
    unwrap_time   bool
       If True, single images are returned instead of image sequences
    output_type  np.dtype
       dtype of generated tensors
    normalize_x  list of tuple
       list the same size as x_img_types containing tuples (scale,offset) used to
       normalize data via   X  -->  (X-offset)*scale.  If None, no scaling is done
    normalize_y  list of tuple
       list the same size as y_img_types containing tuples (scale,offset) used to
       normalize data via   X  -->  (X-offset)*scale

    Returns
    -------
    SEVIRSequence generator

    Examples
    --------

        # Get just Radar image sequences
        vil_seq = SEVIRSequence(x_img_types=['vil'],batch_size=16)
        X = vil_seq.__getitem__(1234)  # returns list the same size as x_img_types passed to constructor

        # Get ir satellite+lightning as X,  radar for Y
        vil_ir_lght_seq = SEVIRSequence(x_img_types=['ir107','lght'],y_img_types=['vil'],batch_size=4)
        X,Y = vil_ir_lght_seq.__getitem__(420)  # X,Y are lists same length as x_img_types and y_img_types

        # Get single images of VIL
        vil_imgs = SEVIRSequence(x_img_types=['vil'], batch_size=256, unwrap_time=True, shuffle=True)

        # Filter out some times
        vis_seq = SEVIRSequence(x_img_types=['vis'],batch_size=32,unwrap_time=True,
                                start_date=datetime.datetime(2018,1,1),
                                end_date=datetime.datetime(2019,1,1),
                                datetime_filter=lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21))

    """

    def __init__(self,
                 tile_list=[1],
                 year_list=['2018'],
                 x_img_types=['vis047','vis086','ir133','sza','insitu'],
                 y_img_types=['insitu'],
                 input_len = 8,
                 output_len = 12,
                 sample_interval = 4,
                 downscale_s = 20,
                 downscale_t = 4,
                 tile_all=DEFAULT_TILELIST,
                 start_date=None,
                 end_date=None,
                 datetime_filter=None,
                 catalog_filter=None,
                 unwrap_time=False,
                 output_type=np.float32,
                 normalize_x=True,
                 normalize_y=False,
                 normalize_max=False,
                 point_based=True,
                 layout='NCHW',
                 batch_size=5,
                 ignorenan=False
                 ):
        self._samples = None
        # self._hdf_files = {}
        self.tile_list = tile_list
        self.year_list = year_list
        self.x_img_types = x_img_types
        self.y_img_types = y_img_types
        self.input_len = input_len
        self.output_len = output_len
        self.sample_interval = sample_interval
        self.downscale_s = downscale_s
        self.downscale_t = downscale_t
        self.tile_all=pd.read_csv(tile_all,low_memory=False)

        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.unwrap_time = unwrap_time
        self.output_type = output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.normalize_max = normalize_max
        self.point_based = point_based
        self.layout = layout
        self.ignorenan = ignorenan
        self.batch_size = batch_size
        self.insitu=pd.read_csv(DEFAULT_INSITU,low_memory=False)
        # if normalize_x:
        #     assert (len(normalize_x) == len(x_img_types))
        # if normalize_y:
        #     assert (len(normalize_y) == len(y_img_types))

        # if self.start_date:
        #     self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        # if self.end_date:
        #     self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        # if self.datetime_filter:
        #     self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        # if self.catalog_filter:
        #     self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._open_index_files()
        self._compute_samples()


    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets
           self._samples  

        """
        # locate all events containing colocated x_img_types and y_img_types
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        imgts = set(imgt)     

        self._samples = self._sample_index_df(imgts=imgts)
        
    def _sample_index_df(self,imgts):
        d = {'tile': [], 'year': [], 'start_index': []}
        sample_size=self.input_len+self.output_len
        for year in self.year_list:
            for tile in self.tile_list:
                start_indices = {}
                for typ in imgts:
                    valid_list = self._index_files[year][tile][typ].tolist()
                    data_array = np.array([x if x > 0 else 0 for x in valid_list])
                    if self.downscale_t:
                        valid_indices = [
                            i for i in range(0, len(data_array) - sample_size + 1, self.sample_interval)
                            if np.any(data_array[i:i + sample_size] > 0)
                        ]
                    else:    
                        valid_indices = [
                            i for i in range(0, len(data_array) - sample_size + 1, self.sample_interval)
                            if np.all(data_array[i:i + sample_size] > 0)
                        ]
                    if self.ignorenan:
                        valid_indices = [
                            i for i in range(0, len(data_array) - sample_size + 1, self.sample_interval)
                        ]
                        
                    start_indices[typ] = set(valid_indices)
                    
                common_indices = sorted(set.intersection(*start_indices.values()))
                num_samples = len(common_indices)
                d['tile'].extend([tile] * num_samples)
                d['year'].extend([year] * num_samples)
                d['start_index'].extend(common_indices)
                print(pd.DataFrame(d))
        return pd.DataFrame(d)
    
    def _open_index_files(self):
        """
        Opens index files
        """
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        index_files = {}
        # print(self.year_list)
        for year in self.year_list:
            index_files[year] = {}  
            for tile in self.tile_list:
                file_path = f'{DEFAULT_DATA_HOME}/{tile}_{year}_index.csv'
                index_files[year][tile] = pd.read_csv(file_path)
        self._index_files = index_files

    def get_batch(self,idx):
        """
        Returns selected batch from the selected SolarSat entries.
        
        Parameters
        ----------
        idx int
          batch index between 0 and __len__()
        return_meta bool
           If true, metadata of samples is also returned. 
           
        Returns
        -------
        If return_meta is False:
           returns X   if self.y_img_types is None
              else returns (X,Y) 
        Elif return_meta is True:
            returns X,meta    if self.y_img_types is None
              else returns (X,Y),meta 
        
        """
        batch = self._get_batch_samples(idx)
        data = {}
        for index, row in batch.iterrows():
            data = self._get_data(row,data)
        t_in=self.input_len
        t_out=self.output_len
        if self.downscale_t:
            t_in = t_in // self.downscale_t
            t_out = t_out // self.downscale_t
        X = [data[t][0, :t_in, :, :] if data[t].ndim == 4 else data[t][:, :t_in]
                for t in self.x_img_types]  # CNTHW format CNT
        # print(len(X))
        
        if self.normalize_x:
            X = [
                self._normalize(X[i], k) if not (self.normalize_max and k == 'ssr') else X[i]  ##will not normalized ssr when it is already normalize_max
                for i, k in enumerate(self.x_img_types)
            ]
        
        X = np.array(X).astype(np.float32)
        
        if np.isnan(X).any==True:
            print('after normalization!:', self.ignorenan)
                    
        X = np.swapaxes(X, 0, 1) # NCTHW format NCT
        # print(X.shape)
        if len(X.shape)==4:
            # print(self.layout)
            X = change_layout_np(X, in_layout='TCHW', out_layout=self.layout)
        # print(X.shape)
        # print('x:', X[0,:,:,0])

        if self.y_img_types:
            # Y = np.array([data[t][0,self.input_len:self.input_len+self.output_len,:,:].astype(np.float32) if data[t].ndim == 4 else data[t][:, self.input_len:self.input_len+self.output_len] for t in self.y_img_types])
            Y = [data[t][0,t_in:t_in+t_out,:,:].astype(np.float32) if data[t].ndim == 4 else data[t][:, t_in:t_in+t_out] for t in self.y_img_types]
            if self.normalize_y:
                Y = [self._normalize(Y[i], k) for i, k in enumerate(self.y_img_types)]
            Y = np.array(Y).astype(np.float32)
            Y = np.swapaxes(Y, 0, 1)
            if len(Y.shape)==4:
                Y = change_layout_np(Y, in_layout='TCHW', out_layout=self.layout)
            # print('y:', Y[0,:,:,0])
            out=X,Y
        else:
            out=X

        # #@
        
        # if return_meta:
        #     meta=self.get_batch_metadata(idx)
        #     return out,meta
        # else:
        return out
        
    def _get_batch_samples(self,idx):
            return self._samples.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
  
    def _genertate_insitu(self, site_id, window_size=15):
        swin = self.insitu[str(site_id)].to_numpy()
        window = np.ones(window_size) / window_size
        half_window = window_size // 2
        # Pad the start and end of the swin array to handle boundary effects
        padded_swin = np.pad(swin, (half_window, half_window), mode='edge')
        # Compute convolution
        convolution = np.convolve(padded_swin, window, mode='valid')
        # Select the means every 15th element starting from the first complete window
        results = convolution[::15]
        return results

    def _get_data(self, row, data):
        """ 
        returns dict { img_type : {"meta" : META, "data": DATA} }
        """
        sample_size=self.input_len+self.output_len
        imgt = self.x_img_types
        if self.y_img_types:
            imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
        imgts = set(imgt)    
        for typ in imgts:
            file_name= '{}/SolarSat_{}_{}_{}.hdf'.format(DEFAULT_DATA_HOME, row['year'], row['tile'], typ)
            if typ == 'insitu':
                idsite = self.tile_all[self.tile_all['tile_id']==row['tile']]['id'].values[0]
                swin = self._genertate_insitu(site_id=idsite)
                data_i = swin[row['start_index']:row['start_index']+sample_size]  
                data_i = data_i[np.newaxis, ...] 
                
                if self.downscale_t:
                    block_size = (1,self.downscale_t)
                    data_i = block_reduce(data_i, block_size, np.nanmean)
                    
                if self.ignorenan:
                    data_i = np.nan_to_num(data_i, nan=0)
                    
                if np.isnan(data_i).any==True:
                    print('in get data:', self.ignorenan)
            else:
                with h5py.File(file_name,'r') as hf:  
                    fillvalue = hf[typ].attrs.get('fillvalue', None)
                    scale_factor = hf[typ].attrs.get('scale_factor', None)
                    array = hf[typ][row['start_index']:row['start_index']+sample_size]   
                    array = array * scale_factor
                    nan_value = fillvalue * scale_factor
                    if typ == 'ssr':
                        array[array==nan_value]=-1      # ocean areas
                    else:
                        array[array==nan_value]=np.nan
                    array[array==nan_value]=np.nan
                    data_i = array[np.newaxis, ...] #(1, T, H, W)
                    # print('hh ',data_i.shape)
                if self.normalize_max:
                    if typ == 'ssr':
                        file_name_sza= '{}/SolarSat_{}_{}_{}.hdf'.format(DEFAULT_DATA_HOME, row['year'], row['tile'], 'sza')
                        with h5py.File(file_name_sza,'r') as hf:  
                            fillvalue = hf['sza'].attrs.get('fillvalue', None)
                            scale_factor = hf['sza'].attrs.get('scale_factor', None)
                            array = hf['sza'][row['start_index']:row['start_index']+sample_size]   
                            array = array * scale_factor
                            nan_value = fillvalue * scale_factor
                            array[array==nan_value]=np.nan
                            data_sza = array[np.newaxis, ...] 
                            data_max = 1360*np.cos(data_sza/180*np.pi)
                            data_i = data_i/data_max
                if self.downscale_s:
                    block_size = (1,1,self.downscale_s,self.downscale_s)
                    data_i = block_reduce(data_i, block_size, np.nanmean)
                if self.downscale_t:
                    # print(self.downscale_t)
                    block_size = (1,self.downscale_t,1,1)
                    data_i = block_reduce(data_i, block_size, np.nanmean)
                if self.ignorenan:
                    data_i = np.nan_to_num(data_i, nan=0)
                if self.point_based:
                    fLine = self.tile_all[self.tile_all['tile_id']==row['tile']]['fLine'].values[0]
                    fCol = self.tile_all[self.tile_all['tile_id']==row['tile']]['fCol'].values[0]
                    fLine = int(fLine / self.downscale_s) if self.downscale_s else fLine
                    fCol = int(fCol / self.downscale_s) if self.downscale_s else fCol
                    data_i = data_i[:, :, fCol, fLine]
                    # print('porint', data_i.shape)
            data[typ] = np.concatenate( (data[typ],data_i),axis=0 ) if (typ in data) else data_i #(1, T, H, W)
            
        return data
    
    def __len__(self):
        """
        How many batches to generate per epoch
        """
        if self._samples is not None:
            # Use floor to avoid sending a batch of < self.batch_size in last batch.   
            max_n = int(np.floor(self._samples.shape[0] / float(self.batch_size)))
        else:
            max_n = 0
        # if self.n_batch_per_epoch is not None:
        #     return min(self.n_batch_per_epoch,max_n)
        # else:
        return max_n
    
    def __getitem__(self, idx):
        """
        Simple wrapper of get_batch that allowed the class to be used as a generator    
        """
        return self.get_batch(idx)  
    
    def _normalize(self, x, typ, reverse=False):
        """
        Normalize data or reverse normalization
        :param x: data array
        :param scale: const scaling value
        :param offset: const offset value
        :param reverse: boolean undo normalization
        :return: normalized x array
        """
        scale_dict = SCALE_SOLARSAT
        offset_dict = OFFSET_SOLARSAT
        
        scale = scale_dict[typ]
        offset = offset_dict[typ]
        if reverse:
            return x * scale + offset
        else:
            return (x-offset) / scale
        
    def preprocess_data_dict(data_dict, data_types=None, layout='NHWT'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
        data_types: Sequence[str]
            The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
        layout: str
            consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
        rescale:    str
            'sevir': use the offsets and scale factors in original implementation.
            '01': scale all values to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
            preprocessed data
        """

        scale_dict = PREPROCESS_SCALE_SOLARSAT
        offset_dict = PREPROCESS_OFFSET_SOLARSAT
        
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = scale_dict[key] * (
                            data.astype(np.float32) +
                            offset_dict[key])
                    data = change_layout_np(data=data,
                                            in_layout='NHWT',
                                            out_layout=layout)
                elif isinstance(data, torch.Tensor):
                    data = scale_dict[key] * (
                            data.float() +
                            offset_dict[key])
                    data = change_layout_torch(data=data,
                                               in_layout='NHWT',
                                               out_layout=layout)
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale='01'):
        """
        Parameters
        ----------
        data_dict
            each data_dict[key] is a torch.Tensor.
        rescale
            str:
                'sevir': data are scaled using the offsets and scale factors in original implementation.
                '01': data are all scaled to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict
            each data_dict[key] is the data processed back in torch.Tensor.
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data = data_dict[key]
            data = data.float() / scale_dict[key] - offset_dict[key]
            data_dict[key] = data
        return data_dict

def save_all_data(dataloader):
    all_x = []
    all_y = []
    
    # Assuming the SOLARSATDataloader supports iteration directly
    for idx in tqdm(range(dataloader.__len__()), desc="Loading data"):
        data = dataloader[idx]
        X, Y = data
        all_x.append(X)
        all_y.append(Y)

    return all_x, all_y  

