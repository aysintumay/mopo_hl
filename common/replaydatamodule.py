import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningDataModule
import os



class ReplayDataModule(LightningDataModule):
    def __init__(self, buffer, batch_size: int = 256, num_workers: int = 11):
        super().__init__()
        self.buffer = buffer  # <- instance of your ReplayBuffer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Convert the full buffer into Tensors
        data = self.buffer.sample_all()
        self.dataset = TensorDataset(
            torch.tensor(data["observations"]).float(),
            torch.tensor(data["actions"]).float(),
            torch.tensor(data["rewards"]).float(),
            torch.tensor(data["next_observations"]).float(),
            torch.tensor(data["terminals"]).float()
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    
    def val_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([0])],  num_workers=self.num_workers,)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([0])],  num_workers=self.num_workers,)
