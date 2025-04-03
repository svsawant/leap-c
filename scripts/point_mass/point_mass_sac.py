"""Main script to run experiments."""

from argparse import ArgumentParser
from pathlib import Path

from leap_c.run import main
from leap_c.rl.sac import SacBaseConfig


parser = ArgumentParser()
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


cfg = SacBaseConfig()
cfg.val.interval = 10_000
cfg.train.steps = 1000_000
cfg.val.num_render_rollouts = 1
cfg.log.wandb_logger = True
cfg.log.tensorboard_logger = True
cfg.sac.entropy_reward_bonus = False  # type: ignore
cfg.sac.update_freq = 4
cfg.sac.batch_size = 64
cfg.sac.lr_pi = 1e-3
cfg.sac.lr_q = 1e-3
cfg.sac.lr_alpha = 1e-3
cfg.sac.init_alpha = 0.10


output_path = Path(f"output/pointmass/sac_{args.seed}")

main("sac", "point_mass", cfg, output_path, args.device)
