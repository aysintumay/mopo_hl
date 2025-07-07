from common.buffer import ReplayBuffer  
import numpy as np
from lightning import LightningDataModule
from omegaconf import ListConfig


class ReplayBufferWrapper(LightningDataModule):
    def __init__(
        self,
        buffer_size: int,
        obs_shape, 
        action_dim: int,
        dataset=None,
        obs_dtype: str = "float32",
        action_dtype: str = "float32",
    ):
        super().__init__()
        if isinstance(obs_shape, (list, tuple, ListConfig)):
            obs_shape = tuple(int(x) for x in obs_shape)  # Convert elements to native int
        else:
            obs_shape = (int(obs_shape),)

        if isinstance(action_dim, (list, tuple, ListConfig)):
            action_dim = int(action_dim[0])
        else:
            action_dim = int(action_dim)
            
        self.buffer = ReplayBuffer(
                        buffer_size=int(buffer_size),
                        obs_shape=obs_shape,
                        obs_dtype=np.dtype(obs_dtype),
                        action_dim=action_dim,
                        action_dtype=np.dtype(action_dtype),
                    )

        self.dataset = dataset

    def load_dataset(self):
        self.buffer.load_dataset(self.dataset)

    def sample(self, batch_size: int):
        return self.buffer.sample(batch_size)

    def get_buffer(self):
        return self.buffer
