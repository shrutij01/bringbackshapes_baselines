from shape_herd.gym_wrappers.twod_playground_env import (
    TwoDPlaygroundEnv,
    NormalizeActions,
)
import dreamerv2.api as dv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=float, default=5e5)
parser.add_argument("--type", type=str, default="dense")
parser.add_argument("--time_limit", type=int, default=1000)
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

dmc_config = dv2.configs["dmc_vision"]
base_config = dv2.defaults.update(dmc_config)
base_config = dv2.defaults.update(
    {
        "logdir": f"./logs/twod_{args.type}/dreamerv2_baseline/seed{args.seed}",
        "task": f"twod_{args.type}",
        "steps": args.steps,
    }
)
config = base_config.parse_flags()

env = TwoDPlaygroundEnv(
    action_repeat=args.action_repeat,
    dense_reward=args.type == "dense",
    time_limit=args.time_limit,
)
env = NormalizeActions(env)

dv2.train(env, config, [dv2.TerminalOutput()])
