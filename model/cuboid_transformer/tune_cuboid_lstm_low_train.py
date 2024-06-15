import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import warnings
from typing import Union, Dict
from shutil import copyfile
from copy import deepcopy
import inspect
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange
from pytorch_lightning import Trainer, seed_everything
from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.utils.checkpoint import pl_ckpt_to_pytorch_state_dict, s3_download_pretrained_ckpt
from earthformer.utils.layout import layout_to_in_out_slice
from earthformer.visualization.nbody import save_example_vis_results
#from earthformer.metrics.sevir import SEVIRSkillScore
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.datasets.LSTM.LSTM_torchlightning_wrap import LSTMLightningDataModule
from earthformer.utils.apex_ddp import ApexDDPStrategy
from pytorch_lightning.utilities.cloud_io import load as pl_load
import time

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")
print(_curr_dir)
pretrained_checkpoints_dir = cfg.pretrained_checkpoints_dir
#pytorch_state_dict_name = "earthformer.pt"

class CuboidLSTMModule(pl.LightningModule):

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super(CuboidLSTMModule, self).__init__()
        self._max_train_iter = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
            dec_use_first_self_attn=model_cfg["dec_use_first_self_attn"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            # global vectors
            num_global_vectors=model_cfg["num_global_vectors"],
            use_dec_self_global=model_cfg["use_dec_self_global"],
            dec_self_update_global=model_cfg["dec_self_update_global"],
            use_dec_cross_global=model_cfg["use_dec_cross_global"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            # initial_downsample
            initial_downsample_type=model_cfg["initial_downsample_type"],
            initial_downsample_activation=model_cfg["initial_downsample_activation"],
            # initial_downsample_type=="conv"
            initial_downsample_scale=model_cfg["initial_downsample_scale"],
            initial_downsample_conv_layers=model_cfg["initial_downsample_conv_layers"],
            final_upsample_conv_layers=model_cfg["final_upsample_conv_layers"],
            # misc
            padding_type=model_cfg["padding_type"],
            z_init_method=model_cfg["z_init_method"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
        )

        self.total_num_steps = total_num_steps
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        self.save_hyperparameters(oc)
        self.oc = oc
        # layout
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        # optimization
        self.max_epochs = oc.optim.max_epochs
        self.optim_method = oc.optim.method
        self.lr = oc.optim.lr
        self.wd = oc.optim.wd
        # lr_scheduler
        self.total_num_steps = total_num_steps
        self.lr_scheduler_mode = oc.optim.lr_scheduler_mode
        self.warmup_percentage = oc.optim.warmup_percentage
        self.min_lr_ratio = oc.optim.min_lr_ratio
        # logging
        self.save_dir = save_dir
        self.logging_prefix = oc.logging.logging_prefix
        # visualization
        self.train_example_data_idx_list = list(oc.vis.train_example_data_idx_list)
        self.val_example_data_idx_list = list(oc.vis.val_example_data_idx_list)
        self.test_example_data_idx_list = list(oc.vis.test_example_data_idx_list)
        self.eval_example_only = oc.vis.eval_example_only

        self.configure_save(cfg_file_path=oc_file)
        # evaluation
        self.metrics_list = oc.dataset.metrics_list
        self.threshold_list = oc.dataset.threshold_list
        self.metrics_mode = oc.dataset.metrics_mode
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        #self.valid_score = SEVIRSkillScore(
        #    mode=self.metrics_mode,
        #    seq_len=self.out_len,
        #    layout=self.layout,
        #    threshold_list=self.threshold_list,
        #    metrics_list=self.metrics_list,
        #    eps=1e-4,)
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        #self.test_score = SEVIRSkillScore(
        #    mode=self.metrics_mode,
        #    seq_len=self.out_len,
        #    layout=self.layout,
        #    threshold_list=self.threshold_list,
        #    metrics_list=self.metrics_list,
        #    eps=1e-4,)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.scores_dir = os.path.join(self.save_dir, 'scores')
        os.makedirs(self.scores_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if (not os.path.exists(cfg_file_target_path)) or \
                    (not os.path.samefile(cfg_file_path, cfg_file_target_path)):
                copyfile(cfg_file_path, cfg_file_target_path)
        self.example_save_dir = os.path.join(self.save_dir, "examples", self.oc.logging.version)
        os.makedirs(self.example_save_dir, exist_ok=True)

    # def get_base_config(self, oc_from_file=None):
    #     oc = OmegaConf.create()
    #     oc.dataset = self.get_dataset_config()
    #     oc.layout = self.get_layout_config()
    #     oc.optim = self.get_optim_config()
    #     oc.logging = self.get_logging_config()
    #     oc.trainer = self.get_trainer_config()
    #     oc.vis = self.get_vis_config()
    #     oc.model = self.get_model_config()
    #     if oc_from_file is not None:
    #         oc = OmegaConf.merge(oc, oc_from_file)
    #     return oc
        
    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        else:
            oc.dataset = self.get_dataset_config()
            oc.layout = self.get_layout_config()
            oc.optim = self.get_optim_config()
            oc.logging = self.get_logging_config()
            oc.trainer = self.get_trainer_config()
            oc.vis = self.get_vis_config()
            oc.model = self.get_model_config()
        return oc

    @staticmethod
    def get_dataset_config():
        oc = OmegaConf.create()
        oc.dataset_name = "lstm"
        oc.img_height = 60
        oc.img_width = 60
        oc.in_len = 12
        oc.out_len = 12
        oc.seq_len = 24
        oc.plot_stride = 2
        oc.interval_real_time = 15
        oc.sample_mode = "sequent"
        oc.stride = oc.out_len
        oc.layout = "NTHWC"
        oc.start_date = None
        oc.train_val_split_date = (2018, 1, 1)
        oc.train_test_split_date = (2018, 6, 1)
        oc.end_date = None
        oc.metrics_mode = "0"
        oc.metrics_list = ('csi', 'pod', 'sucr', 'bias')                   
        oc.threshold_list = (16, 74, 133, 160, 181, 219)                          ##no need
        return oc
    
    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        height = dataset_oc.img_height
        width = dataset_oc.img_width
        in_len = dataset_oc.in_len
        out_len = dataset_oc.out_len
        data_channels = 1
        cfg.input_shape = (in_len, height, width, data_channels)
        cfg.target_shape = (out_len, height, width, data_channels)

        cfg.base_units = 64
        cfg.block_units = None # multiply by 2 when downsampling in each layer
        cfg.scale_alpha = 1.0

        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True

        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"

        cfg.num_global_vectors = 8
        cfg.use_dec_self_global = True
        cfg.dec_self_update_global = True
        cfg.use_dec_cross_global = True
        cfg.use_global_vector_ffn = True
        cfg.use_global_self_attn = False
        cfg.separate_global_qkv = False
        cfg.global_dim_ratio = 1

        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.dec_cross_last_n_frames = None

        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4

        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.pos_embed_type = "t+hw"
        cfg.use_relative_pos = True
        cfg.self_attn_use_final_proj = True
        cfg.dec_use_first_self_attn = False

        cfg.z_init_method = 'zeros'  # The method for initializing the first input of the decoder
        cfg.initial_downsample_type = "conv"
        cfg.initial_downsample_activation = "leaky"
        cfg.initial_downsample_scale = 2
        cfg.initial_downsample_conv_layers = 2
        cfg.final_upsample_conv_layers = 1
        cfg.checkpoint_level = 2
        # initialization
        cfg.attn_linear_init_mode = "0"
        cfg.ffn_linear_init_mode = "0"
        cfg.conv_init_mode = "0"
        cfg.down_up_linear_init_mode = "0"
        cfg.norm_init_mode = "0"
        return cfg

    @classmethod
    def get_layout_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.in_len = dataset_oc.in_len
        oc.out_len = dataset_oc.out_len
        oc.layout = dataset_oc.layout
        return oc

    @staticmethod
    def get_optim_config():
        oc = OmegaConf.create()
        oc.seed = None
        oc.total_batch_size = 32
        oc.micro_batch_size = 8

        oc.method = "adamw"
        oc.lr = 1E-3
        oc.wd = 1E-5
        oc.gradient_clip_val = 1.0
        oc.max_epochs = 50
        # scheduler
        oc.warmup_percentage = 0.2
        oc.lr_scheduler_mode = "cosine"  # Can be strings like 'linear', 'cosine', 'platue'
        oc.min_lr_ratio = 0.1
        oc.warmup_min_lr_ratio = 0.1
        # early stopping
        oc.early_stop = False
        oc.early_stop_mode = "min"
        oc.early_stop_patience = 5
        oc.save_top_k = 1
        return oc

    @staticmethod
    def get_logging_config():
        oc = OmegaConf.create()
        oc.logging_prefix = "LSTM"
        oc.monitor_lr = True
        oc.monitor_device = False
        oc.track_grad_norm = -1
        oc.use_wandb = True
        return oc

    @staticmethod
    def get_trainer_config():
        oc = OmegaConf.create()
        oc.check_val_every_n_epoch = 1
        oc.log_step_ratio = 0.001  # Logging every 1% of the total training steps per epoch
        oc.precision = 32
        return oc

    @classmethod
    def get_vis_config(cls):
        oc = OmegaConf.create()
        dataset_oc = cls.get_dataset_config()
        oc.train_example_data_idx_list = [0, ]
        oc.val_example_data_idx_list = [80, ]
        oc.test_example_data_idx_list = [0, 80, 160, 240, 320, 400]
        oc.eval_example_only = False
        oc.plot_stride = dataset_oc.plot_stride
        return oc

    def configure_optimizers(self):
        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
            'weight_decay': self.oc.optim.wd
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if self.oc.optim.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=self.oc.optim.lr,
                                          weight_decay=self.oc.optim.wd)
        else:
            raise NotImplementedError

        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))

        if self.oc.optim.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.total_num_steps - warmup_iter),
                                                 eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise NotImplementedError
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        r"""
        Default kwargs used when initializing pl.Trainer
        """
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks", [])
        assert isinstance(callbacks, list)
        for ele in callbacks:
            assert isinstance(ele, Callback)
        callbacks += [checkpoint_callback, ]
        if self.oc.logging.monitor_lr:
            callbacks += [LearningRateMonitor(logging_interval='step'), ]
        if self.oc.logging.monitor_device:
            callbacks += [DeviceStatsMonitor(), ]
        if self.oc.optim.early_stop:
            callbacks += [EarlyStopping(monitor="valid_loss_epoch",
                                        min_delta=0.0,
                                        patience=self.oc.optim.early_stop_patience,
                                        verbose=False,
                                        mode=self.oc.optim.early_stop_mode), ]

        logger = kwargs.pop("logger", [])
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.save_dir,version=self.oc.logging.version)
        csv_logger = pl_loggers.CSVLogger(save_dir=self.save_dir,version=self.oc.logging.version)
        logger += [tb_logger, csv_logger]
        if self.oc.logging.use_wandb:
            wandb_logger = pl_loggers.WandbLogger(project=self.oc.logging.logging_prefix,
                                                  save_dir=self.save_dir)
            logger += [wandb_logger, ]

        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        trainer_init_keys = inspect.signature(Trainer).parameters.keys()
        ret = dict(
            callbacks=callbacks,
            # log
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            track_grad_norm=self.oc.logging.track_grad_norm,
            # save
            default_root_dir=self.save_dir,
            # ddp
            accelerator="gpu",
            # strategy="ddp",
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            # optimization
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            # NVIDIA amp
            precision=self.oc.trainer.precision,
        )
        oc_trainer_kwargs = OmegaConf.to_object(self.oc.trainer)
        oc_trainer_kwargs = {key: val for key, val in oc_trainer_kwargs.items() if key in trainer_init_keys}
        ret.update(oc_trainer_kwargs)
        ret.update(kwargs)
        return ret

    @classmethod
    def get_total_num_steps(
            cls,
            num_samples: int,
            total_batch_size: int,
            epoch: int = None):
        r"""
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_lstm_datamodule(dataset_oc,
                             micro_batch_size: int = 1):
        root=dataset_oc['dataset_path']
        name=dataset_oc['dataset_name']
        dm = LSTMLightningDataModule(dataset_oc=dataset_oc, batch_size=micro_batch_size)
        return dm


    def forward(self, batch):
        in_seq, target_seq = batch
        pred_seq = self.torch_nn_module(in_seq)
        loss = F.mse_loss(pred_seq, target_seq)
        return pred_seq, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self(batch)
        self.save_vis_step_end(
            batch_idx=batch_idx,
            in_seq=x, target_seq=y,
            pred_seq=y_hat,
            mode="train"
        )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B, T_out, H, W, C = y.shape
        if not self.eval_example_only or batch_idx in self.val_example_data_idx_list:
            y_hat, _ = self(batch)
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="val"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
            step_mse = self.valid_mse(y_hat, y) * H * W
            step_mae = self.valid_mae(y_hat, y) * H * W
            y_hat = rearrange(y_hat,
                              "b t h w c -> (b t) c h w")
            y = rearrange(y,
                          "b t h w c -> (b t) c h w")

            self.log('valid_mse_step', step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log('valid_mae_step', step_mae, prog_bar=True, on_step=True, on_epoch=False)
        return H, W


    def validation_epoch_end(self, outputs):
        H, W = outputs[0]
        frame_mse = self.valid_mse.compute() * H * W
        frame_mae = self.valid_mae.compute() * H * W
        valid_loss = frame_mse

        self.log('valid_loss_epoch', valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mse_epoch', frame_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mae_epoch', frame_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        print('end')

    def test_step(self, batch, batch_idx):
        x, y = batch
        B, T_out, H, W, C = y.shape
        if not self.eval_example_only or batch_idx in self.test_example_data_idx_list:
            y_hat, _ = self(batch)
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="test"
            )
            if self.precision == 16:
                y_hat = y_hat.float()
            step_mse = self.test_mse(y_hat, y) * H * W
            step_mae = self.test_mae(y_hat, y) * H * W
            y_hat = rearrange(y_hat,
                              "b t h w c -> (b t) c h w")
            y = rearrange(y,
                          "b t h w c -> (b t) c h w")

            self.log('test_mse_step', step_mse, prog_bar=True, on_step=True, on_epoch=False)
            self.log('test_mae_step', step_mae, prog_bar=True, on_step=True, on_epoch=False)
        return H, W

    def test_epoch_end(self, outputs):
        H, W = outputs[0]
        frame_mse = self.test_mse.compute() * H * W
        frame_mae = self.test_mae.compute() * H * W

        self.log('test_mse_epoch', frame_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', frame_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        B, T_out, H, W, C = y.shape
        if not self.eval_example_only or batch_idx in self.test_example_data_idx_list:
            y_hat, _ = self(batch)
            self.save_vis_step_end(
                batch_idx=batch_idx,
                in_seq=x, target_seq=y,
                pred_seq=y_hat,
                mode="test"
            )
            if self.precision == 16:
                y_hat = y_hat.float()

        return y_hat

    def log_score_epoch_end(self, score_dict: Dict, mode: str = "val"):
        if mode == "val":
            log_mode_prefix = "valid"
        elif mode == "test":
            log_mode_prefix = "test"
        else:
            raise ValueError(f"Wrong mode {mode}. Must be 'val' or 'test'.")
        for metrics in self.metrics_list:
            for thresh in self.threshold_list:
                score_mean = np.mean(score_dict[thresh][metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_{thresh}_epoch", score_mean)
            score_avg_mean = score_dict.get("avg", None)
            if score_avg_mean is not None:
                score_avg_mean = np.mean(score_avg_mean[metrics]).item()
                self.log(f"{log_mode_prefix}_{metrics}_avg_epoch", score_avg_mean)

    def save_score_epoch_end(self,
                             score_dict: Dict,
                             mse: Union[np.ndarray, float],
                             mae: Union[np.ndarray, float],
                             mode: str = "val"):
        assert mode in ["val", "test"], f"Wrong mode {mode}. Must be 'val' or 'test'."
        if self.local_rank == 0:
            save_dict = deepcopy(score_dict)
            save_dict.update(dict(mse=mse, mae=mae))
            if self.scores_dir is not None:
                save_path = os.path.join(self.scores_dir, f"{mode}_results_epoch_{self.current_epoch}.pkl")
                f = open(save_path, 'wb')
                pickle.dump(save_dict, f)
                f.close()

    def save_vis_step_end(
            self,
            batch_idx: int,
            in_seq: torch.Tensor, target_seq: torch.Tensor,
            pred_seq: torch.Tensor,
            mode: str = "train"):

        if self.local_rank == 0:
            if mode == "train":
                example_data_idx_list = self.train_example_data_idx_list
            elif mode == "val":
                example_data_idx_list = self.val_example_data_idx_list
            elif mode == "test":
                example_data_idx_list = self.test_example_data_idx_list
            else:
                raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
            if (batch_idx in example_data_idx_list) & (mode != 'train'):
                micro_batch_size = in_seq.shape[self.layout.find("N")]
                data_idx = int(batch_idx * micro_batch_size)
                save_example_vis_results(
                    save_dir=self.example_save_dir,
                    save_prefix=f'{mode}_epoch_{self.current_epoch}_data_{data_idx}',
                    in_seq=in_seq.detach().float().cpu().numpy(),
                    target_seq=target_seq.detach().float().cpu().numpy(),
                    pred_seq=pred_seq.detach().float().cpu().numpy(),
                    layout=self.layout,
                    plot_stride=1,
                    label=self.oc.logging.logging_prefix)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='lstm', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--gpui', default=0, type=int)    #for specific gpu device, but no parallel
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained checkpoints for test.')
    parser.add_argument('--ckpt_name', default=None, type=str,
                        help='The model checkpoint trained on LSTM.')
    return parser

def main():
    start=time.time()
    parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "earthformer_lstm.yaml")) ## currently now pretrained
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        name_prefix= oc_from_file.logging.version+'_'+dataset_oc['dataset_name'][:-4]+'_'+str(dataset_oc['in_len'])+'_'+str(dataset_oc['out_len'])
        print(name_prefix)
    else:
        dataset_oc = OmegaConf.to_object(CuboidLSTMModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
    seed_everything(seed, workers=True)
    dm = CuboidLSTMModule.get_lstm_datamodule(
        dataset_oc=dataset_oc,
        micro_batch_size=micro_batch_size)
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = CuboidLSTMModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
     )
    print('finish1')
    pl_module = CuboidLSTMModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg)
    print('finish2')
    gpuindex=[int(x) for x in str(args.gpui)]
    print(gpuindex)
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=gpuindex, #args.gpus
        accumulate_grad_batches=accumulate_grad_batches,
    )
    print(args.gpus)
    trainer = Trainer(**trainer_kwargs)
    if args.test:
        assert args.ckpt_name is not None, f"args.ckpt_name is required for test!"
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints/tune", args.ckpt_name)
        checkpoint_state=pl_load(ckpt_path, map_location=torch.device("cpu"))
        pl_module.load_state_dict(checkpoint_state)
        pl_module.eval()
        if args.train:
            print('testing training data ...')
            predictions= trainer.predict(model=pl_module,
                     datamodule=dm)
            np.savez_compressed(os.path.join(pl_module.save_dir, "checkpoints/prediction/tune",name_prefix+'_prediction'), predictions)
        else:
            predictions= trainer.test(model=pl_module,
                     datamodule=dm)
            np.savez_compressed(os.path.join(pl_module.save_dir, "checkpoints/prediction/tune",name_prefix+'_test_prediction'), predictions)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        trainer.test(ckpt_path="best",
                     datamodule=dm)
      
        state_dict = pl_load(trainer.checkpoint_callback.best_model_path, map_location=torch.device("cpu"))['state_dict']
        torch.save(state_dict, os.path.join(pl_module.save_dir, "checkpoints/tune", name_prefix+'.pt'))
        predictions=trainer.test(ckpt_path="best",
                     datamodule=dm)
        np.savez_compressed(os.path.join(pl_module.save_dir, "checkpoints/prediction/tune", name_prefix+'_prediction'), predictions)
        end=time.time()
        print(end-start)
        
if __name__ == "__main__":
    main()
