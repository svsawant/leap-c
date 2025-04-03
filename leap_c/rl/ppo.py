from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from statistics import mean
from copy import deepcopy
from collections import defaultdict

from leap_c.nn.mlp import MLP  #, MlpConfig
from leap_c.registry import register_trainer
from leap_c.rl.rollout_buffer import RolloutBuffer
from leap_c.task import Task
from leap_c.trainer import BaseConfig, LogConfig, TrainConfig, Trainer, ValConfig, defaultdict_list
from leap_c.rl.utils import Normal

@dataclass(kw_only=True)
class MlpConfig:
    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"
    weight_init: str | None = "orthogonal"  # If None, no init will be used

@dataclass(kw_only=True)
class PPOAlgorithmConfig:
    """Contains the necessary information for a PPOTrainer.

    Attributes:
        batch_size: The batch size for training.
        buffer_size: The size of the replay buffer.
        gamma: The discount factor.
        tau: The soft update factor.
        soft_update_freq: The frequency of soft updates.
        lr_q: The learning rate for the Q networks.
        lr_pi: The learning rate for the policy network.
        lr_alpha: The learning rate for the temperature parameter.
        num_critics: The number of critic networks.
        report_loss_freq: The frequency of reporting the loss.
        update_freq: The frequency of updating the networks.
    """

    critic_mlp: MlpConfig = field(default_factory=MlpConfig)
    actor_mlp: MlpConfig = field(default_factory=MlpConfig)
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    use_gae: bool = True
    gae_lambda: float = 0.92
    soft_update_freq: int = 1
    lr_v: float = 1e-4
    lr_pi: float = 3e-4
    report_loss_freq: int = 100


@dataclass(kw_only=True)
class PPOBaseConfig(BaseConfig):
    """Contains the necessary information for a Trainer.

    Attributes:
        ppo: The PPO algorithm configuration.
        train: The training configuration.
        val: The validation configuration.
        log: The logging configuration.
        seed: The seed for the trainer.
    """

    ppo: PPOAlgorithmConfig = field(default_factory=PPOAlgorithmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 0


class PPOCritic(nn.Module):
    def __init__(
        self,
        task: Task,
        env: gym.Env,
        mlp_cfg: MlpConfig,
    ):
        super().__init__()

        self.extractor = task.create_extractor(env)
        self.mlp = MLP(
            input_sizes=[self.extractor.output_size],  # type: ignore
            output_sizes=1,
            mlp_cfg=mlp_cfg,
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(self.extractor(x))


class PPOActor(nn.Module):
    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(self, task, env, mlp_cfg: MlpConfig):
        super().__init__()

        self.extractor = task.create_extractor(env)
        action_dim = env.action_space.shape[0]  # type: ignore

        self.mlp = MLP(
            input_sizes=self.extractor.output_size,
            output_sizes=action_dim,  # type: ignore
            mlp_cfg=mlp_cfg,
        )
        self.logstd = nn.Parameter(1.0 * torch.ones(action_dim))
        self.dist_fn = lambda x: Normal(x, self.logstd.exp())

    def forward(self, x: torch.Tensor, a:torch.Tensor=None, deterministic=False):
        e = self.extractor(x)
        dist = self.dist_fn(self.mlp(e))
        log_prob = None
        if a is not None:
            log_prob = dist.log_prob(a)
        return dist, log_prob

    def step(self, x:torch.Tensor, deterministic=False):
        dist, _ = self.forward(x)
        if deterministic:
            a = dist.mode().cpu().detach().numpy()
            logp_a = None
        else:
            a = dist.sample()
            logp_a = dist.log_prob(a).cpu().detach().numpy()
            a = a.cpu().detach().numpy()
        return a, logp_a

@register_trainer("ppo", PPOBaseConfig())
class PPOTrainer(Trainer):
    cfg: PPOBaseConfig

    def __init__(
        self, task: Task, output_path: str | Path, device: str, cfg: PPOBaseConfig
    ):
        """Initializes the trainer with a configuration, output path, and device.

        Args:
            task: The task to be solved by the trainer.
            output_path: The path to the output directory.
            device: The device on which the trainer is running
            cfg: The configuration for the trainer.
        """
        super().__init__(task, output_path, device, cfg)

        self.v = PPOCritic(task, self.train_env, cfg.ppo.critic_mlp)
        self.v_optim = torch.optim.Adam(self.v.parameters(), lr=cfg.ppo.lr_v)

        self.pi = PPOActor(task, self.train_env, cfg.ppo.actor_mlp)  # type: ignore
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.ppo.lr_pi)

        self.basic_buffer = None
        self.buffer = None

        # TODO: Move to def run of main trainer.
        self.to(device)

    def train_loop(self) -> Iterator[int]:
        # is_terminated = is_truncated = True
        # episode_return = episode_length = np.inf

        while True:
            # self.basic_buffer = RolloutBuffer(self.cfg.train.rollout_steps, device=self.device)
            rollouts = RolloutBuffer(
                self.train_env.observation_space,
                self.train_env.action_space,
                self.cfg.train.rollout_steps,
                1
            )

            obs, _ = self.train_env.reset()
            o = self.task.collate([obs], self.device)
            steps = episode_return = episode_length = 0
            is_terminated = is_truncated = False
            stats = defaultdict_list()
            while steps < self.cfg.train.rollout_steps:
                o = self.task.collate([obs], self.device)
                action, log_a = self.pi.step(o)  # type: ignore
                # action = action.cpu().detach().numpy()
                val = self.v.forward(o).cpu().detach().numpy()
                obs_prime, reward, is_terminated, is_truncated, _ = self.train_env.step(action)
                done_flag = is_truncated or is_terminated
                mask = 1 - done_flag

                terminal_val = np.zeros_like(val)
                if is_terminated or is_truncated:
                    terminal_val = self.v.forward(o).cpu().detach().numpy()
                    obs, _ = self.train_env.reset()
                    if episode_length < np.inf:
                        stats["episode_return"].append(episode_return)
                        stats["episode_length"].append(episode_length)
                        # self.report_stats("train", stats, self.state.step)
                    is_terminated = is_truncated = False
                    episode_return = episode_length = 0

                # # TODO (Jasper): Add is_truncated to buffer.
                # self.basic_buffer.put((obs, action, reward, obs_prime, is_terminated,
                #                        log_a, val, terminal_val))  # type: ignore
                rollouts.push({'obs': obs, 'act': action, 'rew': reward, 'mask': mask,
                               'v': val, 'logp': log_a, 'terminal_v': terminal_val})

                obs = obs_prime
                episode_return += float(reward)
                episode_length += 1
                steps += 1

            is_truncated = True if not is_terminated else False
            if is_terminated or is_truncated:
                if episode_length < np.inf:
                    stats["episode_return"].append(episode_return)
                    stats["episode_length"].append(episode_length)
            stats_iter = {"episode_return": mean(stats["episode_return"]), "episode_length": mean(stats["episode_length"])}
            self.report_stats("train", stats_iter, self.state.step)

            # agent training
            last_val = self.v.forward(o).cpu().detach().numpy()
            ret, adv = compute_returns_and_advantages(rollouts.rew,
                                                      rollouts.v,
                                                      rollouts.mask,
                                                      rollouts.terminal_v,
                                                      last_val,
                                                      gamma=self.cfg.ppo.gamma,
                                                      use_gae=self.cfg.ppo.use_gae,
                                                      gae_lambda=self.cfg.ppo.gae_lambda)
            rollouts.ret = ret
            rollouts.adv = (adv-adv.mean()) / (adv.std() + 1e-6)
            loss_stats = self.update(rollouts, self.device)
            self.report_stats("loss", loss_stats, self.state.step+self.cfg.train.rollout_steps)
            yield self.cfg.train.rollout_steps

    def update(self, rollouts, device='cpu'):
        results = defaultdict(list)
        num_mini_batch = rollouts.max_length * rollouts.batch_size // self.cfg.ppo.batch_size
        # assert if num_mini_batch is not 0
        assert num_mini_batch != 0, 'num_mini_batch is 0'
        for _ in range(self.cfg.ppo.opt_epochs):
            p_loss_epoch, v_loss_epoch, e_loss_epoch, kl_epoch = 0, 0, 0, 0
            for batch in rollouts.sampler(self.cfg.ppo.batch_size, device):
                # Actor update.
                policy_loss, entropy_loss, approx_kl = self.compute_policy_loss(batch)
                # Update only when no KL constraint or constraint is satisfied.
                if ((self.cfg.ppo.target_kl <= 0) or
                        (self.cfg.ppo.target_kl > 0 and approx_kl <= 1.5 * self.cfg.ppo.target_kl)):
                    self.pi_optim.zero_grad()
                    (policy_loss + self.cfg.ppo.entropy_coef * entropy_loss).backward()
                    self.pi_optim.step()

                # Critic update.
                value_loss = self.compute_value_loss(batch)
                self.v_optim.zero_grad()
                value_loss.backward()
                self.v_optim.step()

                # logging
                p_loss_epoch += policy_loss.item()
                v_loss_epoch += value_loss.item()
                e_loss_epoch += entropy_loss.item()
                kl_epoch += approx_kl.item()
            results['policy_loss'].append(p_loss_epoch / num_mini_batch)
            results['value_loss'].append(v_loss_epoch / num_mini_batch)
            results['entropy_loss'].append(e_loss_epoch / num_mini_batch)
            results['approx_kl'].append(kl_epoch / num_mini_batch)
        results = {k: sum(v) / len(v) for k, v in results.items()}
        return results

    def compute_policy_loss(self, batch):
        '''Returns policy loss(es) given batch of data.'''
        obs, act, logp_old, adv = batch['obs'], batch['act'], batch['logp'], batch['adv']
        dist, logp = self.pi.forward(obs, act)
        # Policy.
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(
            ratio,
            1 - self.cfg.ppo.clip_param,
            1 + self.cfg.ppo.clip_param
        ) * adv
        policy_loss = -torch.min(ratio * adv, clip_adv).mean()
        # Entropy.
        entropy_loss = -dist.entropy().mean()
        # KL/trust region.
        approx_kl = (logp_old - logp).mean()
        return policy_loss, entropy_loss, approx_kl

    def compute_value_loss(self, batch):
        '''Returns value loss(es) given batch of data.'''
        obs, ret, v_old = batch['obs'], batch['ret'], batch['v']
        v_cur = self.v.forward(obs)
        value_loss = 0.5 * (v_cur - ret).pow(2).mean()
        return value_loss

    def act(
        self, obs, deterministic: bool = False, state=None
    ) -> tuple[np.ndarray, None, None]:
        obs = self.task.collate([obs], self.device)
        with torch.no_grad():
            action, _ = self.pi.step(obs, deterministic=deterministic)
        return action[0], None, None

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return [self.v_optim, self.pi_optim]

    def save(self) -> None:
        """Save the trainer state in a checkpoint folder."""

        torch.save(self.buffer, self.output_path / "buffer.pt")
        return super().save()

    def load(self) -> None:
        """Loads the state of a trainer from the output_path."""

        self.buffer = torch.load(self.output_path / "buffer.pt")
        return super().load()


def compute_returns_and_advantages(rews,
                                   vals,
                                   masks,
                                   terminal_vals=0,
                                   last_val=0,
                                   gamma=0.99,
                                   use_gae=False,
                                   gae_lambda=0.95
                                   ):
    '''Useful for policy-gradient algorithms.'''
    T, N = rews.shape[:2]
    rets, advs = np.zeros((T, N, 1)), np.zeros((T, N, 1))
    ret, adv = last_val, np.zeros((N, 1))
    vals = np.concatenate([vals, last_val[np.newaxis, ...]], 0)
    # Compensate for time truncation.
    rews += gamma * terminal_vals
    # Cumulative discounted sums.
    for i in reversed(range(T)):
        ret = rews[i] + gamma * masks[i] *ret
        if not use_gae:
            adv = ret - vals[i]
        else:
            td_error = rews[i] + gamma * masks[i] * vals[i + 1] - vals[i]
            adv = adv * gae_lambda * gamma * masks[i] + td_error
        rets[i] = deepcopy(ret)
        advs[i] = deepcopy(adv)
    return rets, advs