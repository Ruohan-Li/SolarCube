import numpy as np
import pandas as pd
#######################################################################
# data_path = 'solarsat_image_long_train.npz'
# data = np.load(data_path,allow_pickle=True)
# x = data['arr_0'][:,0,:,:,:,:]
# y = data['arr_1']
# df = data['arr_2']

# print(x.shape)
# print(y.shape)

# y=np.transpose(y,(0,2,1,3,4))
# print(y.shape)#'NCTHW'
# np.savez_compressed(data_path, x,y,df)



#######################################################################
# data_path = 'solarsat_image_long_train.npz'
# data = np.load(data_path,allow_pickle=True)
# x = data['arr_0'].astype(np.float32)
# y = data['arr_1'].astype(np.float32)
# df = data['arr_2']

# print(x.shape)
# print(y.shape)

# np.savez_compressed(data_path, x,y,df)

#######################################################################
data_path = 'solarsat_point_long_test.npz'
data = np.load(data_path,allow_pickle=True)
x = data['arr_0']
y = data['arr_1']
df = data['arr_2']

print(x.shape)
print(y.shape)
# x = np.nan_to_num(x, nan=0)
# y = np.nan_to_num(x, nan=0)

# np.savez_compressed(data_path, x,y,df)