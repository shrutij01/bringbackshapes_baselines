import bringbackshapes
from bringbackshapes.gym_wrappers.envs.bringbackshapes_env import NormalizeActions
import gym
import dreamerv2.api as dv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=float, default=1e6)
parser.add_argument("--type", type=str, default="sparse")
parser.add_argument("--time_limit", type=int, default=1000)
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

dmc_config = dv2.configs["dmc_vision"]
base_config = dv2.defaults.update(dmc_config)
base_config = dv2.defaults.update(
    {
        "logdir": f"./logs/bbs_{args.type}/dreamerv2/seed{args.seed}",
        "task": f"bbs_{args.type}",
        "steps": args.steps,
    }
)
config = base_config.parse_flags()

env = gym.make(
    "bringbackshapes.gym_wrappers:arena-v0",
    action_repeat=args.action_repeat,
    dense_reward=args.type == "dense",
    time_limit=args.time_limit,
)

env = NormalizeActions(env)

dv2.train(env, config, [dv2.TerminalOutput()])
