# src/systems/rl_system.py

import time
import copy
import os
import numpy as np
import torch
from lightning import LightningModule
from tqdm import tqdm
import wandb 
from utils.plotter import plot_q_value, plot_p_loss, plot_accuracy
from torch.utils.data import DataLoader, TensorDataset


class RLSystem(LightningModule):
    def __init__(self, cfg, algo, eval_env, optimizer=None):
        super().__init__()
        self.cfg = cfg
        self.algo = algo
        self.paths = cfg.paths
        # self.logger_model = logger
        self.eval_env = eval_env  # Set later if needed
        self.num_timesteps = 0
        self.optimizer = optimizer
        self.automatic_optimization = False
        # self.ite = cfg.iter

        # For logging
        self.q1_l, self.q2_l, self.q_l = [], [], []
        self.critic_loss1, self.critic_loss2, self.actor_loss = [], [], []
        self.entropy, self.alpha_loss = [], []
        self.reward_l, self.acc_l, self.off_acc = [], [], []
        self.reward_std_l, self.acc_std_l, self.off_acc_std = [], [], []


    def on_fit_start(self):

        if self.cfg.pretrained:
            self.algo.dynamics_model.load_model() 
            print(f"Dynamics model loaded from checkpoint {self.cfg.paths.ckth_path}.")
            
        else:
            start_time = time.time()
            self.algo.learn_dynamics()
            self.algo.save_dynamics_model(f'{self.algo.dynamics_model.model_name}')
            print("Dynamics total time: {:.3f}s".format(time.time() - start_time))


    def training_step(self, batch, batch_idx):
        #batch, batch_idx are not used not only for structural reasons
        if self.num_timesteps % self.cfg.rl_model.rollout_freq == 0:
            self.algo.rollout_transitions()

        loss, q_values = self.algo.learn_policy()

        # wandb_logger = self.trainer.logger
        # if isinstance(wandb_logger, wandb.sdk.wandb_run.Run):  # optional safety check
        #     wandb_logger = self.trainer.logger 

        self.log("loss/actor", loss["loss/actor"], prog_bar=True)
        self.log("loss/critic1", loss["loss/critic1"], prog_bar=True)
        self.log("q_value/q1", q_values["q1"], prog_bar=True)

        self.q1_l.append(q_values["q1"])
        self.q2_l.append(q_values["q2"])
        self.q_l.append(q_values["q_target"])

        self.critic_loss1.append(loss["loss/critic1"])
        self.critic_loss2.append(loss["loss/critic2"])
        self.actor_loss.append(loss["loss/actor"])
        self.entropy.append(loss["entropy"])
        self.alpha_loss.append(loss["loss/alpha"])

        # self.log("loss/critic1", loss["loss/critic1"])
        # self.log("loss/critic2", loss["loss/critic2"])
        # self.log("loss/actor", loss["loss/actor"])
        # self.log("entropy", loss["entropy"])
        # self.log("loss/alpha", loss["loss/alpha"])
        # self.log("q1", q_values["q1"])
        # self.log("q2", q_values["q2"])
        # self.log("q_target", q_values["q_target"])

        self.num_timesteps += 1
        # return loss["loss/actor"]

    def validation_step(self, batch, batch_idx):
        print('validation triggered')
        eval_info = self._evaluate()
        avg_reward = np.mean(eval_info["eval/episode_reward"])
        avg_length = np.mean(eval_info["eval/episode_length"])

        wandb_logger = self.trainer.logger
        if isinstance(wandb_logger, wandb.sdk.wandb_run.Run):  # optional safety check
            wandb_logger = self.trainer.logger 
        # Log to Lightning
        wandb_logger.experiment.log({
        "eval/avg_episode_reward": avg_reward,
        "eval/avg_episode_length": avg_length,
        "global_step": self.global_step
                })
        self.log("eval/avg_episode_reward", avg_reward, prog_bar=True)
        self.log("eval/avg_episode_length", avg_length, prog_bar=True)
        self.log("global_step", self.global_step, prog_bar=True)

        # return {"reward": avg_reward, "length": avg_length}


    def on_train_end(self):
        # Save policy
        model_save_dir = self.paths.ckth_path
        #change this later
        os.makedirs(model_save_dir, exist_ok=True)
        policy_copy = copy.deepcopy(self.algo.policy)
        torch.save(policy_copy.to("cpu").state_dict(),
                   os.path.join(model_save_dir, f"policy_{self.cfg.env.task}.pth"))
        
        wandb_logger = self.trainer.logger
        if isinstance(wandb_logger, wandb.sdk.wandb_run.Run):  # optional safety check
            wandb_logger = self.trainer.logger

        def save_wandb_plot(name, fig):
            wandb_logger.experiment.log({f"{name} Value": wandb.Image(fig)})
        # Plot
        # if wandb_logger.id != 0:
        fig = plot_q_value(np.array(self.q1_l).reshape(-1, 1), 'Q1')
        save_wandb_plot('Q1', fig)
        fig = plot_q_value(np.array(self.q2_l).reshape(-1, 1), 'Q2')
        save_wandb_plot('Q2', fig)
        fig = plot_q_value(np.array(self.q_l).reshape(-1, 1), 'Q')
        save_wandb_plot('Q', fig)

        fig = plot_p_loss(np.array(self.critic_loss1).reshape(-1, 1), 'Critic1')
        save_wandb_plot('Critic1', fig)
        fig = plot_p_loss(np.array(self.critic_loss2).reshape(-1, 1), 'Critic2')
        save_wandb_plot('Critic2', fig)
        fig = plot_p_loss(np.array(self.actor_loss).reshape(-1, 1), 'Actor')
        save_wandb_plot('Actor', fig)
        fig = plot_p_loss(np.array(self.entropy).reshape(-1, 1), 'Entropy')
        save_wandb_plot('Entropy', fig)
        fig = plot_p_loss(np.array(self.alpha_loss).reshape(-1, 1), 'Alpha')
        save_wandb_plot('Alpha', fig)

            # fig = plot_accuracy(np.array(self.reward_l), np.array(self.reward_std_l) / self.cfg.eval_episodes, 'Return')
            # self.logger.log({f"{'Return'} Value": wandb.Image(fig)})
            # fig = plot_accuracy(np.array(self.acc_l), np.array(self.acc_std_l) / self.cfg.eval_episodes, 'Accuracy')
            # self.logger.log({f"{'Accuracy'} Value": wandb.Image(fig)})
            # fig = plot_accuracy(np.array(self.off_acc), np.array(self.off_acc_std) / self.cfg.eval_episodes, '1-off Accuracy')
            # self.logger.log({f"{'1-off Accuracy'} Value": wandb.Image(fig)})

        print("Policy total training time complete.")

   
    def configure_optimizers(self):
        return self.optimizer
    
    # def test_step(self, batch, batch_idx) -> None:
    #     """Perform a single test step on a batch of data from the test set.

    #     :param batch: A batch of data (a tuple) containing the input tensor of images and target
    #         labels.
    #     :param batch_idx: The index of the current batch.
    #     """
    #     loss, preds, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.test_loss(loss)
    #     self.test_acc(preds, targets)
    #     self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        print('test_step triggered')
        eval_info = self._evaluate()
        avg_reward = np.mean(eval_info["eval/episode_reward"])
        avg_length = np.mean(eval_info["eval/episode_length"])

        wandb_logger = self.trainer.logger
        if isinstance(wandb_logger, wandb.sdk.wandb_run.Run):  # optional safety check
            wandb_logger = self.trainer.logger 
        # Log to Lightning
        wandb_logger.experiment.log({
        "eval/avg_episode_reward": avg_reward,
        "eval/avg_episode_length": avg_length,
        "global_step": self.global_step
                })
        self.log("eval/avg_episode_reward", avg_reward, prog_bar=True)
        self.log("eval/avg_episode_length", avg_length, prog_bar=True)
        self.log("global_step", self.global_step, prog_bar=True)


    # def on_test_epoch_end(self) -> None:
    #     """Lightning hook that is called when a test epoch ends."""
    #     print('test_end triggered')
    #     eval_info = self._evaluate()
    #     avg_reward = np.mean(eval_info["eval/episode_reward"])
    #     avg_length = np.mean(eval_info["eval/episode_length"])

    #     wandb_logger = self.trainer.logger
    #     if isinstance(wandb_logger, wandb.sdk.wandb_run.Run):  # optional safety check
    #         wandb_logger = self.trainer.logger 
    #     # Log to Lightning
    #     wandb_logger.experiment.log({
    #     "eval/avg_episode_reward": avg_reward,
    #     "eval/avg_episode_length": avg_length,
    #     "global_step": self.global_step
    #             })
    #     self.log("eval/avg_episode_reward", avg_reward, prog_bar=True)
    #     self.log("eval/avg_episode_length", avg_length, prog_bar=True)
    #     self.log("global_step", self.global_step, prog_bar=True)
    #     pass
    
    @torch.no_grad()
    def _evaluate(self):
        self.algo.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self.cfg.train.eval_episodes:
            action = self.algo.policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )

                #d4rl don't have REF_MIN_SCORE and REF_MAX_SCORE for v2 environments
                dset_name = self.eval_env.unwrapped.spec.name+'-v0'
                # print(d4rl.get_normalized_score(dset_name, np.array(episode_reward))*100)

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([0])])
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([0])])
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader([torch.tensor([0])])


    
