import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib
import random
import numpy as np
import torch
import gym
import d4rl

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

import hydra
from omegaconf import DictConfig
# from common.buffer import ReplayBuffer
from common.replaydatamodule import ReplayDataModule
# from common.replay_buffer_wrapper import ReplayBufferWrapper

from systems.rl_system.rl_systems import RLSystem
from systems.builder import build_policy

from algos.mopo import MOPO


from utils.instantiators import (
   
    instantiate_callbacks,
    instantiate_loggers,
 
)
from utils.pylogger import RankedLogger
from utils.logging_utils import log_hyperparameters

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(config_path="../configs_ai", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    env = gym.make(cfg.dataset)
    dataset = d4rl.qlearning_dataset(env)

    # dataset = {key: dataset[key][:10] for key in dataset.keys()}
    env.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.trainer.accelerator != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    task_name = cfg.env.task.split('-')[0]
    static_fns = importlib.import_module(f"static_fns.{task_name}").StaticFns
    device = f'cuda:{cfg.train.device}' if torch.cuda.is_available() else 'cpu'
    policy, actor_optim, critic1_optim, critic2_optim = build_policy(cfg, env)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if cfg.model._target_ == "src.models.transition_model.TransitionModel":
        dynamics_model: LightningModule = hydra.utils.instantiate(cfg.model,
            obs_space=env.observation_space,
            action_space=env.action_space,
            static_fns=static_fns,
            device=device,
            paths = cfg.paths,
            **cfg.model
        )
    # TODO: worl model throws nan error after the frist MOPO epoch
    elif cfg.model._target_ == "src.models.world_model.D4RLWorldModel":
        dynamics_model: LightningModule = hydra.utils.instantiate(cfg.model,
            env_name=cfg.dataset,
            device=device,
            paths = cfg.paths,
            **cfg.model
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model._target_}")

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape


    log.info(f"Instantiating data buffer <{cfg.buffer.data._target_}>")
    offline_buffer_wrapper: LightningDataModule = hydra.utils.instantiate(cfg.buffer.data, 
                                                                          obs_shape=obs_shape,
                                                                            action_dim=action_dim,
                                                                            dataset=dataset)
    # offline_buffer_wrapper.dataset = d4rl.qlearning_dataset(env)
    offline_buffer_wrapper.load_dataset()
    # offline_buffer_wrapper.load_dataset(dataset)
    offline_buffer = offline_buffer_wrapper.get_buffer()
    datamodule = ReplayDataModule(
                buffer=offline_buffer_wrapper.get_buffer(),
                batch_size=cfg.train.batch_size,
                )
    
    # cfg.buffer.model.obs_shape = obs_shape
    # cfg.buffer.model.action_dim = action_dim
    log.info(f"Instantiating model buffer <{cfg.buffer.model._target_}>")
    offline_modelbuffer_wrapper: LightningDataModule = hydra.utils.instantiate(cfg.buffer.model, 
                                                                               obs_shape=obs_shape, 
                                                                                action_dim=action_dim,
                                                                                dataset=dataset,
                                                                                )
    model_buffer = offline_modelbuffer_wrapper.get_buffer()
    
 
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)


    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[WandbLogger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.rl_model._target_}>")
    algo: LightningModule = hydra.utils.instantiate(
                    cfg.rl_model,
                    policy=policy,
                    dynamics_model=dynamics_model,
                    offline_buffer=offline_buffer,
                    model_buffer=model_buffer,
                    # rollout_length=cfg.rl_model.rollout_length,
                    batch_size=cfg.train.batch_size,
                    # real_ratio=cfg.rl_model.real_ratio,
                    log_fn=logger[0].experiment,  # or logger.experiment if using single WandbLogger
                    **cfg.rl_model
                )
   

    system = RLSystem(cfg=cfg, algo=algo, eval_env = env, optimizer = [actor_optim, critic1_optim, critic2_optim])
    log.info("started the modules")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "env": env,

        "databuffer": offline_buffer,
        "modelbuffer": model_buffer, 
        "model": algo,
        "policy": policy,
        "transition_model": dynamics_model,

        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=system, datamodule = datamodule, ckpt_path=None)
        trainer.test(model=system, datamodule = datamodule, ckpt_path=None)


    # trainer = pl.Trainer(
    #     max_epochs=cfg.train.max_epochs,
    #     accelerator="gpu" if cfg.device == "cuda" else "cpu",
    #     devices=1,
    #     logger=wandb_logger,
    #     log_every_n_steps=50,
    # )
    # trainer.fit(system)

if __name__ == "__main__":
    main()