import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import callbacks
import stable_baselines3
from shape_herd.gym_wrappers.twod_playground_env import (
    TwoDPlaygroundEnv,
    NormalizeActions,
)
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=float, default=1e6)
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--logdir", type=str, default="./logs")
parser.add_argument("--id", type=str, default="default")
parser.add_argument("--type", type=str, default="dense")
parser.add_argument("--time_limit", type=int, default=3000)
parser.add_argument("--action_repeat", type=int, default=4)
parser.add_argument("--nenvs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

base_log_dir = args.logdir
log_name = f"twod_{args.type}/{args.id}/seed{args.seed}"
log_dir = os.path.join(base_log_dir, log_name)

env = TwoDPlaygroundEnv(
    action_repeat=args.action_repeat,
    dense_reward=args.type == "dense",
    time_limit=args.time_limit,
)
env = NormalizeActions(env)

nenvs = args.nenvs if args.algo != "SAC" else 1
env = make_vec_env(lambda: env, n_envs=nenvs)
env = VecCheckNan(env, raise_exception=True)

checkpoint_callback = CheckpointCallback(100000, log_dir)
eval_env = TwoDPlaygroundEnv(
    action_repeat=args.action_repeat,
    dense_reward=args.type == "dense",
    time_limit=args.time_limit,
)
eval_env = NormalizeActions(eval_env)
save_video_callback = callbacks.SaveVideoCallback(
    eval_env=eval_env, eval_freq=1000, log_dir=log_dir
)
callback = CallbackList([checkpoint_callback, save_video_callback])

algo_class = getattr(stable_baselines3, args.algo)
model = algo_class(
    "CnnPolicy", env, verbose=1, tensorboard_log=base_log_dir, seed=args.seed
)

model.learn(
    total_timesteps=args.steps, callback=callback, tb_log_name=f"{log_name}/"
)
