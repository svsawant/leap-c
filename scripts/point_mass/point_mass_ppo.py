"""Main script to run experiments."""

from argparse import ArgumentParser
from pathlib import Path

from leap_c.run import main
from leap_c.rl.ppo import PPOBaseConfig


parser = ArgumentParser()
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


cfg = PPOBaseConfig()
cfg.val.interval = 50_000
cfg.train.rollout_steps = 1000
cfg.train.steps = 1000_000
cfg.val.num_render_rollouts = 0
cfg.log.wandb_logger = True
cfg.log.tensorboard_logger = True
# cfg.ppo.entropy_reward_bonus = False  # type: ignore
# cfg.ppo.update_freq = 4
cfg.ppo.opt_epochs = 20
cfg.ppo.clip_param = 0.2
cfg.ppo.target_kl = 0.02
cfg.ppo.entropy_coef = 0.0001
cfg.ppo.batch_size = 128
cfg.ppo.lr_pi = 3e-3
cfg.ppo.lr_v = 3e-3


output_path = Path(f"output/pointmass/ppo_{args.seed}")

main("ppo", "point_mass", cfg, output_path, args.device)
