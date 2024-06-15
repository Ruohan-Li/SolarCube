import os
import numpy as np
import pandas as pd
import h5py
from skimage.measure import block_reduce
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
TYPES    = ['vis047','vis086','ir133','dsr','sza','insitu']
DEFAULT_DATA_HOME = os.path.abspath(os.path.join( '..', '..', 'data', 'geonex_sat'))
DEFAULT_TILELIST   = DEFAULT_DATA_HOME + '/solarsat_sitelist.csv'
DEFAULT_INSITU = DEFAULT_DATA_HOME+'/solarsat_insitu.csv'

# TODO : MANY!

# Example function of how to create pytorch dataloader with sevir dataset
def get_solarsat_loader(batch_size=3,
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
                     solarsat_data_home=DEFAULT_DATA_HOME,
                     shuffle=False,
                     pin_memory=True,
                     output_type=np.float32,
                     normalize_x=None,
                     normalize_y=None,
                     normalize_max=True,
                     num_workers=4,
                     ):
    dataset = SOLARSAT(
                    tile_list,
                    year_list,
                    x_img_types,
                    y_img_types,
                    input_len,
                    output_len,
                    sample_interval,
                    downscale,
                    tile_all,
                    start_date,
                    end_date,
                    datetime_filter,
                    catalog_filter,
                    unwrap_time,
                    solarsat_data_home,
                    output_type,
                    normalize_x,
                    normalize_y,
                    normalize_max)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      num_workers=num_workers)


class SOLARSAT(Dataset):
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
                 batch_size=1
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

        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.unwrap_time = unwrap_time
        self.output_type = output_type
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.normalize_max = normalize_max
        self.batch_size = batch_size
        self.insitu=pd.read_csv(DEFAULT_INSITU,low_memory=False)
        if normalize_x:
            assert (len(normalize_x) == len(x_img_types))
        if normalize_y:
            assert (len(normalize_y) == len(y_img_types))

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

    # def _compute_samples(self):
    #     """
    #     Computes the list of samples in catalog to be used. This sets
    #        self._samples

    #     """
    #     # locate all events containing colocated x_img_types and y_img_types
    #     imgt = self.x_img_types
    #     if self.y_img_types:
    #         imgt = list(set(imgt + self.y_img_types))  # remove duplicates
    #     imgts = set(imgt)
    #     filtcat = self.catalog[np.logical_or.reduce([self.catalog.img_type == i for i in imgt])]
    #     # remove rows missing one or more requested img_types
    #     filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
    #     # If there are repeated IDs, remove them (this is a bug in SEVIR)
    #     filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0] == len(imgt))
    #     self._samples = filtcat.groupby('id').apply(lambda df: df_to_series(df, imgt, unwrap_time=self.unwrap_time))

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
        print(self.year_list)
        for year in self.year_list:
            index_files[year] = {}  
            for tile in self.tile_list:
                file_path = f'{DEFAULT_DATA_HOME}/{tile}_{year}_index.csv'
                index_files[year][tile] = pd.read_csv(file_path)
        self._index_files = index_files

    # def _open_files(self):
    #     """
    #     Opens HDF files
    #     """
    #     imgt = self.x_img_types
    #     if self.y_img_types:
    #         imgt = list(set(imgt + self.y_img_types))  # remove duplicates
    #     hdf_filenames = []
    #     for t in imgt:
    #         hdf_filenames += list(np.unique(self._samples[f'{t}_filename'].values))
    #     self._hdf_files = {}
    #     for f in hdf_filenames:
    #         print('Opening HDF5 file for reading', f)
    #         self._hdf_files[f] = h5py.File(self.sevir_data_home + '/' + f, 'r')

    def __getitem__(self, idx):
        """
        Simple wrapper of get_batch that allowed the class to be used as a generator    
        """
        return self.get_batch(idx,return_meta=False)  
    
    # def __getitem__(self, idx):
    #     """
    #     batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

    #     return np.array([
    #         resize(imread(file_name), (200, 200))
    #            for file_name in batch_x]), np.array(batch_y)
    #     """
    #     data = {}
    #     data = read_data(self._samples.iloc[idx], data, self._hdf_files, self.unwrap_time)
    #     X = [data[t].astype(self.output_type) for t in self.x_img_types]
    #     if self.normalize_x:
    #         X = [normalize(X[k], s[0], s[1]) for k, s in enumerate(self.normalize_x)]
    #     if self.y_img_types is not None:
    #         Y = [data[t].astype(self.output_type) for t in self.y_img_types]
    #         if self.normalize_y:
    #             Y = [normalize(Y[k], s[0], s[1]) for k, s in enumerate(self.normalize_y)]
    #         return X, Y
    #     else:
    #         return X
    
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
            data = self._get_data(row,data)
        # print(len(data['ssr']))
        # print(data['ssr'].shape)
        # X = [data[t].astype(self.output_type) for t in self.x_img_types]
        X = np.array([data[t][0, :self.input_len, :, :].astype(np.float32) if data[t].ndim == 4 else data[t][:, :self.input_len] for t in self.x_img_types])  #BCTHW
        if self.normalize_x:
            X = [normalize(X[k],s) for k,s in enumerate(self.normalize_x)]

        if self.y_img_types is not None:
            Y = np.array([data[t][0,self.input_len:self.input_len+self.output_len,:,:].astype(np.float32) if data[t].ndim == 4 else data[t][:, self.input_len:self.input_len+self.output_len] for t in self.y_img_types])
            # Y = [data[t] for t in self.y_img_types]
            if self.normalize_y:
                Y = [normalize(Y[k],s) for k,s in enumerate(self.normalize_y)]
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
  
    # def get_batch_metadata(self,idx):
    #     """
    #     Returns the SEVIR metadata for batch index
    #     """
    #     # return only these cols (independent of img_type)
    #     cols = ['id','time_utc','episode_id','event_id','event_type','minute_offsets',
    #             'llcrnrlat','llcrnrlon','urcrnrlat','urcrnrlon','proj','height_m','width_m']            
    #     batch = self._get_batch_samples(idx)
    #     imgtyps = np.unique([x.split('_')[0] for x in list(batch.keys())])
    #     meta=[]
    #     for k,i in enumerate([v[0] for v in batch.index.values]):
    #         m = self.catalog[self.catalog.id==i].iloc[0][cols]
    #         if self.unwrap_time: # adjust time to exact time of image
    #             m['time_utc']+=pd.Timedelta(seconds=FRAME_TIMES[batch.iloc[k][f'{imgtyps[0]}_time_index']])
    #             m.pop('minute_offsets')
    #         meta.append(m)
    #     return pd.DataFrame(meta)
    
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
            data[typ] = np.concatenate( (data[typ],data_i),axis=0 ) if (typ in data) else data_i
            
        # print(data)
        return data
    
# def read_data(row, data, hdf_files, unwrap_time=False):
#     """
#     Reads data from data object
#     :param row: series with fields IMGTYPE_filename IMGTYPE_index, IMGTYPE_time_index
#     :param data: data object
#     :param hdf_files: hdf_file handles to read from
#     :param unwrap_time: boolean for unwrapping time field
#     :return:
#     """
#     image_types = np.unique([x.split('_')[0] for x in list(row.keys())])
#     for t in image_types:
#         f_name = row[f'{t}_filename']
#         idx = row[f'{t}_index']
#         if unwrap_time:
#             time_idx = row[f'{t}_time_index']
#             t_slice = slice(time_idx, time_idx + 1)
#         else:
#             t_slice = slice(0, None)
#         # Need to bin lght counts into grid
#         if t == 'lght':
#             lightning_data = hdf_files[f_name][idx][:]
#             data_i = lightning_to_hist(lightning_data, t_slice)
#         else:
#             data_i = hdf_files[f_name][t][idx:idx + 1, :, :, t_slice]
#         data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i
#     return data

# def lightning_to_hist(data, t_slice=slice(0, None)):
#     """
#     Converts Nx5 lightning data matrix into a XYT histogram
#     :param data: lightning event data
#     :param t_slice: temporal dimension
#     :return: XYT histogram of lightning data
#     """

#     out_size = (48, 48, len(FRAME_TIMES)) if t_slice.stop is None else (48, 48, 1)
#     if data.shape[0] == 0:
#         return np.zeros((1,) + out_size, dtype=np.float32)

#     # filter out points outside the grid
#     x, y = data[:, 3], data[:, 4]
#     m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
#     data = data[m, :]
#     if data.shape[0] == 0:
#         return np.zeros((1,) + out_size, dtype=np.float32)

#     # Filter/separate times
#     t = data[:, 0]
#     if t_slice.stop is not None:  # select only one time bin
#         if t_slice.stop > 0:
#             tm = np.logical_and(t >= FRAME_TIMES[t_slice.stop - 1],
#                                 t < FRAME_TIMES[t_slice.stop])
#         else:  # special case:  frame 0 uses lightning from frame 1
#             tm = np.logical_and(t >= FRAME_TIMES[0], t < FRAME_TIMES[1])
#         data = data[tm, :]
#         z = np.zeros(data.shape[0], dtype=np.int64)
#     else:  # compute z coordinate based on bin location times
#         z = np.digitize(t, FRAME_TIMES) - 1
#         z[z == -1] = 0  # special case:  frame 0 uses lightning from frame 1

#     x = data[:, 3].astype(np.int64)
#     y = data[:, 4].astype(np.int64)

#     k = np.ravel_multi_index(np.array([y, x, z]), out_size)
#     n = np.bincount(k, minlength=np.prod(out_size))
#     return np.reshape(n, out_size).astype(np.float32)[np.newaxis, :]


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
        return x / scale + offset
    else:
        return (x-offset) * scale


# def df_to_series(df, image_types, n_frames=49, unwrap_time=False):
#     """
#     This looks like it takes a data frame and turns it into a sequence like data frame
#     :param df: pandas data frame
#     :param image_types: image types to extract
#     :param n_frames: number of frames to extract
#     :param unwrap_time: boolean for whether or not to set time index
#     :return: sequence data frame
#     """
#     d = {}
#     df = df.set_index('img_type')
#     for i in image_types:
#         s = df.loc[i]
#         idx = s.file_index if i != 'lght' else s.id
#         if unwrap_time:
#             d.update({f'{i}_filename': [s.file_name] * n_frames,
#                       f'{i}_index': [idx] * n_frames,
#                       f'{i}_time_index': range(n_frames)})
#         else:
#             d.update({f'{i}_filename': [s.file_name],
#                       f'{i}_index': [idx]})
#     return pd.DataFrame(d)