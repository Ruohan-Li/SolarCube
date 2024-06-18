import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import sys
print(sys.path)
from solarcube.solarsat_torch_wrap import SolarSatDataModule
import sys
from tqdm import tqdm
import numpy as np
import argparse
from omegaconf import OmegaConf
torch.multiprocessing.set_sharing_strategy('file_system')

def memory_stats(device):
    print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MiB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f} MiB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved(device)/1024**2:.2f} MiB")


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('--save_path',default=None, type=str)

args = parser.parse_args()
oc_from_file = OmegaConf.load(open(args.cfg, "r"))
dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
random_seed = oc_from_file.optim.seed
name_prefix=str(dataset_oc['train_tile_list'])+'_'+str(dataset_oc['test_tile_list'])+'_'+str(dataset_oc['x_img_types'])+str(dataset_oc['y_img_types'])
print(name_prefix)
        
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.cuda.set_device(1)  

dm = SolarSatDataModule(dataset_oc=dataset_oc)
dm.prepare_data()
dm.setup()

trainLoader = dm.train_dataloader()
validLoader = dm.val_dataloader()
testLoader = dm.test_dataloader()


def data():
    trainvalLoader = dm.trainval_dataloader()
    print('saving test data...')
    x_test=[]
    y_test=[]
    t = tqdm(testLoader, leave=False, total=len(testLoader))
    for i, ( inputVar,targetVar) in enumerate(t):
        # Check for NaNs in inputVar and targetVar before appending
        if torch.isnan(inputVar).any():
            print(f"NaN detected in inputVar at batch {i}")
        if torch.isnan(targetVar).any():
            print(f"NaN detected in targetVar at batch {i}")
        
        x_test.append(inputVar.detach().cpu())  # Ensure tensors are detached and moved to CPU
        y_test.append(targetVar.detach().cpu())
    x_test_tensor = torch.cat(x_test, dim=0)  # Adjust 'dim' as necessary for your data
    y_test_tensor = torch.cat(y_test, dim=0)
    x_test_np = x_test_tensor.numpy()
    y_test_np = y_test_tensor.numpy()
    if np.isnan(x_test_np).any():
        print('final!!')
    print(x_test_np.shape, y_test_np.shape)
    np.savez_compressed(args.save_path+'_test.npz', x_test_np, y_test_np, dm.lstm_test.solarsat_dataloader._samples)

    print('saving train data...')
    x_train=[]
    y_train=[]
    t = tqdm(trainvalLoader, leave=False, total=len(trainvalLoader))
    for i, ( inputVar,targetVar) in enumerate(t):
        x_train.append(inputVar.detach().cpu())  # Ensure tensors are detached and moved to CPU
        y_train.append(targetVar.detach().cpu())
    x_train_tensor = torch.cat(x_train, dim=0)  # Adjust 'dim' as necessary for your data
    y_train_tensor = torch.cat(y_train, dim=0)
    x_train_np = x_train_tensor.numpy()
    y_train_np = y_train_tensor.numpy()
    print(x_train_np.shape, y_train_np.shape)
    np.savez_compressed(args.save_path+'_train.npz', x_train_np, y_train_np, dm.lstm_train_val.solarsat_dataloader._samples)
    
if __name__ == "__main__":
    print('loading data...')
    data()

