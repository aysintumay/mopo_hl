import torch
import numpy as np

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algos.sac import SACPolicy


def build_policy(cfg, env):
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    obs_shape = obs_shape[0]
    print("obs_shape", obs_shape)
    print("action_dim", action_dim)

    # actor + critic networks
    actor_backbone = MLP(input_dim=obs_shape, hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=obs_shape + action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=obs_shape + action_dim, hidden_dims=[256, 256])

    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim", 256),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
    )

    actor = ActorProb(actor_backbone, dist, cfg.train.device)
    critic1 = Critic(critic1_backbone, cfg.train.device)
    critic2 = Critic(critic2_backbone, cfg.train.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.policy.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfg.policy.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfg.policy.critic_lr)

    # entropy tuning
    if cfg.policy.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=cfg.train.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=cfg.policy.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

       

    policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=cfg.policy.tau,
        gamma=cfg.policy.gamma,
        alpha=alpha,
        device=cfg.train.device
    )
    return policy, actor_optim, critic1_optim, critic2_optim


# def build_model(cfg, env, static_fns):
#     model = TransitionModel(
#         obs_space=env.observation_space,
#         action_space=env.action_space,
#         static_fns=static_fns,
#         lr=cfg.model.dynamics_lr,
#         device=cfg.trainer.devices,
#         **cfg
#     )
#     return model
