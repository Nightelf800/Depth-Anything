# Copyright (c) 2023 42dot. All rights reserved.

import torch
from torch.utils.data import DataLoader

from VFDepth.dataset import construct_dataset


_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']


class VFDepthAlgo:
    """
    Model class for "Self-supervised surround-view depth estimation with volumetric feature fusion"
    """

    def __init__(self, config, ddad_cfg, rank):
        super(VFDepthAlgo, self).__init__()
        self.rank = rank
        self.mode = 'train'
        self.dataloaders = {}
        self.read_config(ddad_cfg)
        self.prepare_dataset(config, ddad_cfg, rank)


    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)


    def prepare_dataset(self, config, ddad_cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')

        if self.mode == 'train':
            self.set_train_dataloader(config, ddad_cfg, rank)
            self.set_val_dataloader(config, ddad_cfg)

        if self.mode == 'eval':
            self.set_eval_dataloader(config, ddad_cfg)

    def set_train_dataloader(self, config, ddad_cfg, rank):
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (config.input_height, config.input_width),
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(ddad_cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': config.batch_size,
            'shuffle': True,
            'num_workers': config.workers,
            'pin_memory': True,
            'drop_last': True
        }

        if config.distributed:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                shuffle=True
            )
            dataloader_opts['sampler'] = self.train_sampler

        self.dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // (config.batch_size * config.world_size) * config.epochs

    def set_val_dataloader(self, config, ddad_cfg):
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(config.input_height), int(config.input_width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        val_dataset = construct_dataset(ddad_cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self.dataloaders['val'] = DataLoader(val_dataset, **dataloader_opts)

    def set_eval_dataloader(self, config, ddad_cfg):
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(config.height), int(config.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(ddad_cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': config.batch_size,
            'shuffle': False,
            'num_workers': config.workers,
            'pin_memory': True,
            'drop_last': True
        }

        self.dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)


    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']




