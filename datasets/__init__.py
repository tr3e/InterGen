import lightning.pytorch as pl
import torch
from .interhuman import InterHumanDataset
from datasets.evaluator import (
    EvaluatorModelWrapper,
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader)
# from .dataloader import build_dataloader

__all__ = [
    'InterHumanDataset', 'EvaluationDataset',
    'get_dataset_motion_loader', 'get_motion_loader']

def build_loader(cfg, data_cfg):
    # setup data
    if data_cfg.NAME == "interhuman":
        train_dataset = InterHumanDataset(data_cfg)
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        )

    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "interhuman":
            self.train_dataset = InterHumanDataset(self.cfg)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            )
