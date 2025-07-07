import argparse
import datetime
import os
import random
import importlib
import wandb 
import pickle

import gym
import d4rl
import envs.abiomed_env as abiomed_env
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer,plot_accuracy
from common.util import set_device_and_logger
from common import util




def get_eval(policy, world_model,  env, logger, trainer, args, data):


    trainer.eval_env = env
    trainer.algo.policy = policy

    eval_info, dataset = trainer._evaluate(data, world_model, args)


    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    logger.print(f"episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    logger.record("eval/episode_reward", ep_reward_mean, args.eval_episodes, printed=False)
   

    dset_name = env.unwrapped.spec.name+'-v0'
    normalized_score_mean = d4rl.get_normalized_score(dset_name, ep_reward_mean)*100
    normalized_score_std = d4rl.get_normalized_score(dset_name, ep_reward_std)*100
    logger.record("normalized_episode_reward", normalized_score_mean, ep_length_mean, printed=False)
    logger.print(f"normalized_episode_reward: {normalized_score_mean:.3f} ± {normalized_score_std:.3f},\
                        episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    
    return dataset, eval_info



def test(i, args, world_model, model_logger, sac_policy, trainer, offline_buffer=None, log_path=None):


    log_path = os.path.join(log_path, 'test',  f'ite_{i}', args.mode)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    # Devid = args.devid if args.device == 'cuda' else -1
    device_model = set_device_and_logger(args.device.index,logger, model_logger)
    # args.device = device_model

    # create env and dataset
   
    
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)

    
    env.seed(args.seed)


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    # policy_state_dict = torch.load(os.path.join(model_logger.log_path, f'policy_{args.task}.pth'))
    # sac_policy.load_state_dict(policy_state_dict)

    test_dataset, eval_info = get_eval(sac_policy, world_model, env, logger, trainer, args, offline_buffer)

    return test_dataset, eval_info

if __name__ == "__main__":


    
    test()
