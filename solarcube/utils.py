"""
Input generator for sevir
"""

import os
import numpy as np
import pandas as pd
import h5py
import math
from skimage.measure import block_reduce
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

# List all available types
TYPES    = ['vis047','vis086','ir133','dsr','sza','insitu']

import pathlib
_thisdir = str(pathlib.Path(__file__).parent.absolute())
DEFAULT_DATA_HOME = os.path.abspath(os.path.join( '..', '..', 'data', 'geonex_sat'))
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarsat_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarsat_insitu.csv'
# DEFAULT_FILE_NAME = 

# Nominal Frame time offsets in minutes (used for non-raster types)

# NOTE:  The lightning flashes in each from will represent the 5 minutes leading up the
# the frame's time EXCEPT for the first frame, which will use the same flashes as the second frame
#  (This will be corrected in a future version of SEVIR so that all frames are consistent)
FRAME_TIMES = np.arange(-120.0,125.0,5) * 60 # in seconds

# # Record dtypes for reading
# DTYPES={'vis047':np.uint8,
#         'vis086':np.int16,
#         'ir133':np.int16,
#         'dsr':np.int16,
#         'insitu':np.int16}


class SolarSatSequence():
    """
    Class for creating training and testing dataset from SolarSat.  This class can be used directly for streaming data into
    model, however this is a bit slow due to file IO.  It is recommended to use this class to first prepare
    a dataset for training.
    
    Parameters
    ----------
    catalog  str or pd.DataFrame
        name of SolarSat site file to be read in, or an already read in and processed catalog
    x_img_types  list 
        List of image types to be used as model inputs.  For types, run SSolarSatSequence.get_types()
    y_img_types  list or None
       List of image types to be used as model targets (if None, __getitem__ returns only x_img_types )
    solarsat_data_home  str
       Directory path to SolarSat data
    catalog  str
       Name of SolarSat site CSV file.  
    batch_size  int
       batch size to generate
    n_batch_per_epoch  int or None
       Number of batches in an epoch.  Set to None to match available data
    start_date   datetime
       Start time of SolarSat samples to generate   
    end_date    datetime
       End time of SolarSat samples to generate
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
    shuffle  bool
       If True, data samples are shuffled before each epoch
    shuffle_seed   int
       Seed to use for shuffling
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
    SolarSatSequence object
    
    Examples
    --------
    
        # Get ir satellite+lightning as X,  radar for Y
        solar_seq = SolarSatSequence(tile_list=['h15v03','h55v15'],x_img_types=['vis047','ir133'],y_img_types=['ssr'],batch_size=4)
        X,Y = solar_seq.__getitem__(20)  # X,Y are numpy array in shape (C,T,W,H), C is same as the length of x_img_types and y_img_types
        
        # Filter out some times
        vis_seq = SolarSatSequence(x_img_types=['vis'],batch_size=32,unwrap_time=True,
                                 start_date=datetime.datetime(2018,1,1),
                                 end_date=datetime.datetime(2019,1,1),
                                 datetime_filter=lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21))
    
    """
    def __init__(self,
                 tile_list=['h15v03'],
                 year_list=['2018'],
                 x_img_types=['vil'],
                 y_img_types=None, 
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
                 shuffle=False,
                 shuffle_seed=1,
                 output_type=np.float32,
                 normalize_x=None,
                 normalize_y=None,
                 normalize_max=True,
                 verbose=False,
                 batch_size = 3,
                 point_based = False,
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
        self.downscale = downscale
        self.tile_all=pd.read_csv(tile_all,low_memory=False)

        self.batch_size=batch_size
        self.datetime_filter=datetime_filter
        self.catalog_filter=catalog_filter
        self.start_date=start_date
        self.end_date=end_date
        self.unwrap_time = unwrap_time
        self.shuffle=shuffle
        self.shuffle_seed=int(shuffle_seed)
        self.output_type=output_type
        self.normalize_max = normalize_max
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.verbose=verbose
        self.insitu=pd.read_csv(DEFAULT_INSITU,low_memory=False)
        self.point_based=point_based
        if normalize_x:
            assert(len(normalize_x)==len(x_img_types))
        if normalize_y:
            assert(len(normalize_y)==len(y_img_types))

        # if self.start_date:
        #     self.tile_list = self.tile_list[self.tile_list.time_utc > self.start_date ]
        # if self.end_date:
        #     self.tile_list = self.tile_list[self.tile_list.time_utc <= self.end_date]
        # if self.datetime_filter:
        #     self.tile_list = self.tile_list[self.datetime_filter(self.tile_list.time_utc)]
        
        # if self.catalog_filter:
        #     self.tile_list = self.tile_list[self.catalog_filter(self.tile_list)]
        
        self._open_index_files()
        self._compute_samples()
        
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
        if self.shuffle:
            self._samples=self._samples.sample(frac=1,random_state=self.shuffle_seed)
    
    def _sample_index_df(self,imgts):
        d = {'tile': [], 'year': [], 'start_index': []}
        sample_size=self.input_len+self.output_len
        for year in self.year_list:
            for tile in self.tile_list:
                start_indices = {}
                for typ in imgts:
                    valid_list = self._index_files[year][tile][typ].tolist()
                    data_array = np.array([x if x > 0 else 0 for x in valid_list])
                    valid_indices = [
                        i for i in range(0, len(data_array) - sample_size + 1, self.sample_interval)
                        if np.all(data_array[i:i + sample_size] > 0)
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
        for year in self.year_list:
            index_files[year] = {}  
            for tile in self.tile_list:
                file_path = f'{DEFAULT_DATA_HOME}/{tile}_{year}_index.csv'
                index_files[year][tile] = pd.read_csv(file_path)
        self._index_files = index_files
        
    def __getitem__(self, idx):
        """
        Simple wrapper of get_batch that allowed the class to be used as a generator    
        """
        return self.get_batch(idx,return_meta=False)  
    
    def get_batch(self,idx, return_meta=False):
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
            data = self._get_data(row,data) #BTHW
        # X = [data[t].astype(self.output_type) for t in self.x_img_types]
        X = np.stack([data[t][:, :self.input_len, :, :].astype(np.float32) if data[t].ndim == 4 else data[t][:, :self.input_len] for t in self.x_img_types], axis=-1)  
        #BTHWC
        if self.normalize_x:
            X = [SolarSatSequence.normalize(X[k],s) for k,s in enumerate(self.normalize_x)]

        if self.y_img_types is not None:
            Y = np.stack([data[t][:,self.input_len:self.input_len+self.output_len,:,:].astype(np.float32) if data[t].ndim == 4 else data[t][:, self.input_len:self.input_len+self.output_len] for t in self.y_img_types], axis=-1)
            # Y = [data[t] for t in self.y_img_types]
            if self.normalize_y:
                Y = [SolarSatSequence.normalize(Y[k],s) for k,s in enumerate(self.normalize_y)]
            out=X,Y
        else:
            out=X

        #@
        
        if return_meta:
            meta=self.get_batch_metadata(idx)
            return out,meta
        else:
            return out
    
    def _get_batch_samples(self,idx):
        return self._samples.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
    

    def get_batch_metadata(self,idx):
        """
        Returns the SEVIR metadata for batch index
        """
        # return only these cols (independent of img_type)
        cols = ['id','time_utc','episode_id','event_id','event_type','minute_offsets',
                'llcrnrlat','llcrnrlon','urcrnrlat','urcrnrlon','proj','height_m','width_m']            
        batch = self._get_batch_samples(idx)
        imgtyps = np.unique([x.split('_')[0] for x in list(batch.keys())])
        meta=[]
        for k,i in enumerate([v[0] for v in batch.index.values]):
            m = self.catalog[self.catalog.id==i].iloc[0][cols]
            if self.unwrap_time: # adjust time to exact time of image
                m['time_utc']+=pd.Timedelta(seconds=FRAME_TIMES[batch.iloc[k][f'{imgtyps[0]}_time_index']])
                m.pop('minute_offsets')
            meta.append(m)
        return pd.DataFrame(meta)
    
        
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
                idsite = self.tile_all[self.tile_all['tile_id']==row['tile']]['id'][0]
                swin = self._genertate_insitu(site_id=idsite)
                data_i = swin[row['start_index']:row['start_index']+sample_size]  
                data_i = data_i[np.newaxis, ...] 
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
                    data_i = array[np.newaxis, ...] 
                if self.normalize_max:
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
                if self.downscale is not None:
                    block_size = (1,1,self.downscale,self.downscale)
                    data_i = block_reduce(data_i, block_size, np.mean)
                if self.point_based:
                    fLine = self.tile_all[self.tile_all['tile_id']==row['tile']]['fLine'][0]
                    fCol = self.tile_all[self.tile_all['tile_id']==row['tile']]['fCol'][0]
                    fLine = int(fLine / self.downscale) if self.downscale else fLine
                    fCol = int(fCol / self.downscale) if self.downscale else fCol
                    data_i = data_i[:, :, fCol, fLine]
            data[typ] = np.concatenate( (data[typ],data_i),axis=0 ) if (typ in data) else data_i
            
        # print(data)
        return data

    def load_batches(self,
                     n_batches=10,
                     offset=0,
                     progress_bar=False,
                     return_meta=False):
        """
        Loads a selected number of batches into memory.  This returns the concatenated
        result of [self.__getitem__(i+offset) for i in range(n_batches)]

        WARNING:  Be careful about running out of memory.

        Parameters
        ----------
        n_batches   int
            Number of batches to load.   Set to -1 to load them all, but becareful
            not to run out of memory
        offset int
            batch offset to apply
        progress_bar  bool
            Show a progress bar during loading (requires tqdm module)
        return_meta bool
            If true, returns metadata in addition to images.  See self.get_batch

        """
        if progress_bar:
            try:
                from tqdm import tqdm as RW
            except ImportError:
                print('You need to install tqdm to use progress bar')
                RW=list
        else:
            RW=list
        
        n_batches = self.__len__() if n_batches==-1 else n_batches
        n_batches = min(n_batches,self.__len__())
        assert(n_batches>0)
        
        def out_shape(n_batches,shp,batch_size):
            """
            Computes shape for preinitialization
            """
            return (n_batches*batch_size,*shp)

        bidx=0
        if self.y_img_types is None: # one output
            X=None
            meta=None
            for i in RW( range(offset,offset+n_batches) ):
                Xi = self.get_batch(i,return_meta=False)
                if X is None:
                    shps = [out_shape(n_batches,xi.shape[1:],xi.shape[0]) for xi in Xi] 
                    #X = [np.empty( s,dtype=DTYPES[k] ) for s,k in zip(shps,self.x_img_types)]
                    X = [np.empty(s) for s in shps]
                for ii,xi in enumerate(Xi):
                    X[ii][bidx:bidx+xi.shape[0]] = xi
                bidx+=xi.shape[0]
                if return_meta:
                    meta_i=self.get_batch_metadata(i)
                    meta = meta_i if meta is None else pd.concat((meta,meta_i)) 
            if return_meta:
                return X,meta
            else:
                return X
                
        else:
            X,Y=None,None
            meta=None
            for i in RW( range(offset,offset+n_batches) ):
                Xi,Yi = self.get_batch(i,return_meta=False)
                if X is None:
                    shps_x = [out_shape(n_batches,xi.shape[1:],xi.shape[0]) for xi in Xi]
                    shps_y = [out_shape(n_batches,yi.shape[1:],yi.shape[0]) for yi in Yi]
                    # X = [np.empty(s,dtype=DTYPES[k]) for s,k in zip(shps_x,self.x_img_types)]
                    # Y = [np.empty(s,dtype=DTYPES[k]) for s,k in zip(shps_y,self.y_img_types)]
                    X = [np.empty(s) for s in shps_x]
                    Y = [np.empty(s) for s in shps_y]
                for ii,xi in enumerate(Xi):
                    X[ii][bidx:bidx+xi.shape[0]] = xi
                for ii,yi in enumerate(Yi):
                    Y[ii][bidx:bidx+yi.shape[0]] = yi   
                bidx+=xi.shape[0]
                if return_meta:
                    meta_i=self.get_batch_metadata(i)
                    meta = meta_i if meta is None else pd.concat((meta,meta_i)) 
            if return_meta:
                return (X,Y),meta
            else:
                return X,Y

    def on_epoch_end(self):
        """
        Used e.g. in tensorflow datagenerators.  Shuffles rows after epoch ends
        """
        if self.shuffle:
            self._samples.sample(frac=1,random_state=self.shuffle_seed)
    
    # def close(self):
    #     """
    #     Closes all open file handles
    #     """
    #     for f in self._hdf_files:
    #         self._hdf_files[f].close()
    #     self._hdf_files={}

    # def __del__(self):
    #     for f,hf in self._hdf_files.items():
    #         try:
    #             hf.close()
    #         except ImportError:
    #             pass # okay when python shutting down
        
        
    # def _read_data(self,row,data):
    #     """
    #     row is a series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index
    #     """
    #     imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
    #     for t in imgtyps:
    #         fname = row[f'{t}_filename']
    #         idx   = row[f'{t}_index']
    #         #t_slice = row[f'{t}_time_index'] if self.unwrap_time else slice(0,None)
    #         if self.unwrap_time:
    #             tidx=row[f'{t}_time_index']
    #             t_slice = slice(tidx,tidx+1) 
    #         else:
    #             t_slice = slice(0,None)
    #         # Need to bin lght counts into grid
    #         if t=='lght':
    #             lght_data = self._hdf_files[fname][idx][:]
    #             data_i = self._lght_to_grid(lght_data,t_slice)
    #         else:
    #             data_i = self._hdf_files[fname][t][idx:idx+1,:,:,t_slice]
    #         data[t] = np.concatenate( (data[t],data_i),axis=0 ) if (t in data) else data_i
    #     print(data)
            
    #     return data
    
    

    # def _open_files(self,verbose=True):
    #     """
    #     Opens HDF files
    #     """
    #     imgt = self.x_img_types
    #     if self.y_img_types:
    #         imgt=list( set(imgt + self.y_img_types) ) # remove duplicates
    #     hdf_filenames = []
    #     for t in imgt:
    #         hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))
    #     self._hdf_files = {}
    #     for f in hdf_filenames:
    #         if verbose:
    #             print('Opening HDF5 file for reading',f)
    #         self._hdf_files[f] = h5py.File(self.sevir_data_home+'/'+f,'r')

    # def save(self,filename):
    #     """
    #     Saves generator to a file for easier reloading
    #     """
    #     self.close()
    #     pickle.dump(open(filename,'wb'))
    #     self._open_files(verbose=False)
    
    @staticmethod
    def get_types():
        return TYPES
    
    @staticmethod
    def normalize(X,s):
        """
        Normalized data using s = (scale,offset) via Z = (X-offset)*scale
        """
        return (X-s[1])*s[0]

    @staticmethod
    def unnormalize(Z,s):
        """
        Reverses the normalization performed in a SolarSatSequence generator
        given s=(scale,offset)
        """
        return Z/s[0]+s[1]
    
    
 
def normalize(x, scale, offset, reverse=False):
    """
    Normalize data or reverse normalization
    :param x: data array
    :param scale: const scaling value
    :param offset: const offset value
    :param reverse: boolean undo normalization
    :return: normalized x array
    """
    if reverse:
        return x * scale + offset
    else:
        return (x-offset) / scale
    
def change_layout_np(data,
                     in_layout='NCTHW', out_layout='NTCHW',
                     ret_contiguous=False):
    # first convert to 'NCTHW'
    if len(data.shape)==5:
        if in_layout == 'NCTHW':
            pass
        elif in_layout == 'CNTHW':
            data = np.transpose(data,
                                axes=(1, 0, 2, 3, 4))
        else:
            raise NotImplementedError

        if out_layout == 'NCTHW':
            pass
        elif out_layout == 'NTCHW':
            data = np.transpose(data,
                                axes=(0, 2, 1, 3, 4))
        elif out_layout == 'NTHWC':
            data = np.transpose(data,
                                axes=(0, 2, 3, 4, 1))
        else:
            raise NotImplementedError
        if ret_contiguous:
            data = data.ascontiguousarray()
    if len(data.shape)==4:
        if in_layout == 'CTHW':
            pass
        elif in_layout == 'TCHW':
            data = np.transpose(data,
                                axes=(1, 0, 2, 3))
        else:
            raise NotImplementedError

        if out_layout == 'CTHW':
            pass
        elif out_layout == 'TCHW':
            data = np.transpose(data,
                                axes=(1,0,2,3))
        elif out_layout == 'THWC':
            data = np.transpose(data,
                                axes=(1,2,3,0))
        else:
            raise NotImplementedError
        if ret_contiguous:
            data = data.ascontiguousarray()
    return data

def generate_time_all(year=2018, interval=15):
    # Start and end dates for the year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31 23:"+'{0:02}'.format(60-interval)

    # Generate datetime range with 15-minute intervals
    dt_range = pd.date_range(start=start_date, end=end_date, freq='{0:02}'.format(interval)+'T')

    # Convert to 'YYYYDDDHHMM' format
    #time_data = dt_range.strftime('%Y%j%H%M').astype(np.int32)  # %j is day of year as a zero-padded decimal number

    return dt_range

def genertate_time(tile_id, start, year=2018, interval=15):
    dt_range = generate_time_all(year, interval)    
    utc_time = dt_range[start]
    
    sitelist = pd.read_csv(DEFAULT_TILELIST)
    sites = sitelist.groupby('tile_id').get_group(tile_id)
    meta = sites[0:1].squeeze()
    dt_offset = pd.Timedelta(hours=meta['time_offset'])
    local_time = utc_time + dt_offset

    return utc_time, local_time