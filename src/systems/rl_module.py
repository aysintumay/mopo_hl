import pytorch_lightning as pl
import torch
from src.systems.builder import build_policy, build_model

class RLSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.policy = build_policy(cfg.model)
        # Add replay buffer, logger, etc. here

    def training_step(self, batch, batch_idx):
        loss = self.model.update(batch)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
