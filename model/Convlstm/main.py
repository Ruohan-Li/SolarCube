#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
# print(sys.path)
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params
# from data.SolarSat_torch_wrap import SolarSatDataModule
from solarsat.solarsat_torch_wrap import SolarSatDataModule
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from thop import profile
from omegaconf import OmegaConf
torch.multiprocessing.set_sharing_strategy('file_system')

def print_memory_usage():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f'Allocated memory: {allocated / 1024**3:.2f} GB')
    print(f'Cached memory: {cached / 1024**3:.2f} GB')
    
def memory_stats(device):
    print(f"Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MiB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MiB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f} MiB")
    print(f"Max memory reserved: {torch.cuda.max_memory_reserved(device)/1024**2:.2f} MiB")

def get_best_checkpoint(save_dir):
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in the directory.")
    
    best_checkpoint = sorted(checkpoints, key=lambda x: float(x.split('_')[-1].replace('.pth.tar', '')))[0]
    return os.path.join(save_dir, best_checkpoint)

parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--cfg', default=None, type=str)
# parser.add_argument('-cgru',
#                     '--convgru',
#                     help='use convgru as base cell',
#                     action='store_true')
# parser.add_argument('--file',
#                     type=str)
parser.add_argument('--directory',
                    type=str)
parser.add_argument('--test',
                    action='store_true')
parser.add_argument('--flops',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
# parser.add_argument('-frames_input',
#                     default=8,
#                     type=int,
#                     help='sum of input frames')
# parser.add_argument('-frames_output',
#                     default=12,
#                     type=int,
#                     help='sum of predict frames')
parser.add_argument('--data',
                    action='store_true')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
args = parser.parse_args()
oc_from_file = OmegaConf.load(open(args.cfg, "r"))
dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
total_batch_size = oc_from_file.optim.total_batch_size
micro_batch_size = oc_from_file.optim.micro_batch_size
max_epochs = oc_from_file.optim.max_epochs
random_seed = oc_from_file.optim.seed
# name_prefix=dataset_oc['dataset_source']+'_'+dataset_oc['time_res']+'_'+str(dataset_oc['in_len'])+'_'+str(dataset_oc['out_len'])+'_'
name_prefix=str(dataset_oc['train_tile_list'])+'_'+str(dataset_oc['test_tile_list'])+'_'+str(dataset_oc['x_img_types'])+str(dataset_oc['y_img_types'])
print(name_prefix)
        
TIMESTAMP = args.directory
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.cuda.set_device(1)  

save_dir = './convlstm_save_model/' + TIMESTAMP
save_dir_validate = './testing/' + TIMESTAMP
if not os.path.isdir(save_dir_validate):
        os.makedirs(save_dir_validate)
if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
dm = SolarSatDataModule(dataset_oc=dataset_oc)
dm.prepare_data()
dm.setup()

trainLoader = dm.train_dataloader()
validLoader = dm.val_dataloader()
testLoader = dm.test_dataloader()


if args.convlstm:
    print('convlstm')
    # encoder_params = convlstm_encoder_params(image_size=dataset_oc['img_width'], seq_len=dataset_oc['input_len'])
    # decoder_params = convlstm_decoder_params(image_size=dataset_oc['img_width'], seq_len=dataset_oc['output_len'])
    encoder_params = convlstm_encoder_params(image_size=120, seq_len=dataset_oc['input_len'])
    decoder_params = convlstm_decoder_params(image_size=120, seq_len=dataset_oc['output_len'])
# if args.convgru:
#     encoder_params = convgru_encoder_params
#     decoder_params = convgru_decoder_params
# else:
#     encoder_params = convgru_encoder_params
#     decoder_params = convgru_decoder_params
def data():
    trainvalLoader = dm.trainval_dataloader()
    print('saving test data...')
    x_test=[]
    y_test=[]
    t = tqdm(testLoader, leave=False, total=len(testLoader))
    for i, ( inputVar,targetVar) in enumerate(t):
        x_test.append(inputVar.detach().cpu())  # Ensure tensors are detached and moved to CPU
        y_test.append(targetVar.detach().cpu())
    x_test_tensor = torch.cat(x_test, dim=0)  # Adjust 'dim' as necessary for your data
    y_test_tensor = torch.cat(y_test, dim=0)
    x_test_np = x_test_tensor.numpy()
    y_test_np = y_test_tensor.numpy()
    print(x_test_np.shape, y_test_np.shape)
    np.savez_compressed('solarsat_point_if_test.npz', x_test_np, y_test_np, dm.lstm_test.solarsat_dataloader._samples)

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
    np.savez_compressed('solarsat_point_if_train.npz', x_train_np, y_train_np, dm.lstm_train_val.solarsat_dataloader._samples)
    
def test():
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], seq_len=dataset_oc['output_len']).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    print(os.path.join(save_dir, 'checkpoint.pth.tar'))

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        predictions=list()
        x_test=[]
        y_test=[]
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
        with torch.no_grad():
            net.eval()
            t = tqdm(testLoader, leave=False, total=len(testLoader))
            for i, ( inputVar,targetVar) in enumerate(t):
                x_test.append(inputVar)
                y_test.append(targetVar)
                inputs = inputVar.to(device)
                # label = targetVar.to(device)
                pred = net(inputs)
                predictions.append(pred.cpu().data.numpy())
            x_test=np.concatenate(x_test, axis=0)
            x_test=x_test.reshape(-1, *x_test.shape[-4:])
            x_test=x_test.transpose((0,1,3,4,2))
            y_test=np.concatenate(y_test, axis=0)
            y_test=y_test.reshape(-1, *y_test.shape[-4:])
            y_test=y_test.transpose((0,1,3,4,2))
            predictions=np.concatenate(predictions, axis=0) #B,T,C,H,W
            predictions=predictions.reshape(-1, *predictions.shape[-4:])
            predictions=predictions.transpose((0,1,3,4,2))
        np.savez_compressed(os.path.join(save_dir_validate, TIMESTAMP+'_cnnlstm_prediction'), predictions)
        
        np.savez_compressed(os.path.join(save_dir_validate, TIMESTAMP+'_input'), x_test,y_test,dm.lstm_test.solarsat_dataloader._samples)
    else:
        print('no such checkpoint')

def flops():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    memory_stats(device)
    
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], seq_len=dataset_oc.output_len).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)

    net.to(device)
    memory_stats(device)

    print(os.path.join(save_dir, 'checkpoint.pth.tar'))

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        predictions=list()
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        input_data, _ = next(iter(testLoader))
        input_data = input_data.to(device)
        flops, params = profile(net, inputs=(input_data,))
        print(f"FLOPs: {flops}, Params: {params}")
        # optimizer = torch.optim.Adam(net.parameters())
        # optimizer.load_state_dict(model_info['optimizer'])
        # cur_epoch = model_info['epoch'] + 1
        # with torch.no_grad():
        #     net.eval()
        #     t = tqdm(testLoader, leave=False, total=len(testLoader))
        #     for i, ( inputVar,targetVar) in enumerate(t):
        #         inputs = inputVar.to(device)
        #         label = targetVar.to(device)
        #         pred = net(inputs)
        #         predictions.append(pred.cpu().data.numpy())
        #     predictions=np.array(predictions)
        #     predictions=predictions.reshape(-1, *predictions.shape[-4:])
        #     predictions=predictions.transpose((0,1,3,4,2))
        #     print(predictions.shape)
        # np.savez_compressed(os.path.join(save_dir, args.file[:-4]+'_'+str(args.frames_input)+'_'+str(args.frames_output)+'_prediction'), predictions)
    else:
        print('no such checkpoint')
        input_data, _ = next(iter(testLoader))
        input_data = input_data.to(device)
        flops, params = profile(net, inputs=(input_data,))
        print(f"FLOPs: {flops}, Params: {params}")

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], seq_len=dataset_oc['output_len']).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        print(t)
        for i, ( inputVar,targetVar) in enumerate(t):
            print(inputVar.shape, targetVar.shape)
            inputs = inputVar.to(device).type(torch.cuda.FloatTensor)  # B,S,C,H,W
            label = targetVar.to(device).type(torch.cuda.FloatTensor)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, ( inputVar,targetVar) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device).type(torch.cuda.FloatTensor)
                label = targetVar.to(device).type(torch.cuda.FloatTensor)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
            
    best_checkpoint_path = get_best_checkpoint(save_dir)
    model_info = torch.load(best_checkpoint_path)
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])

    with torch.no_grad():
        net.eval()
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, ( inputVar,targetVar) in enumerate(t):
            x_test.append(inputVar)
            y_test.append(targetVar)
            inputs = inputVar.to(device)
            # label = targetVar.to(device)
            pred = net(inputs)
            predictions.append(pred.cpu().data.numpy())
        x_test=np.concatenate(x_test, axis=0)
        x_test=x_test.reshape(-1, *x_test.shape[-4:])
        x_test=x_test.transpose((0,1,3,4,2))
        y_test=np.concatenate(y_test, axis=0)
        y_test=y_test.reshape(-1, *y_test.shape[-4:])
        y_test=y_test.transpose((0,1,3,4,2))
        predictions=np.concatenate(predictions, axis=0) #B,T,C,H,W
        predictions=predictions.reshape(-1, *predictions.shape[-4:])
        predictions=predictions.transpose((0,1,3,4,2))
    np.savez_compressed(os.path.join(save_dir_validate, TIMESTAMP+'_cnnlstm_prediction'), predictions)

    # Predict using the best model
    with torch.no_grad():
        predictions = model(test_data)


if __name__ == "__main__":
    if args.test:
        print('predicting...')
        test()
    elif args.flops:
        print('calculating flops...')
        flops()
    elif args.data:
        print('loading data...')
        data()
    else:
        train()
