"""
A very basic arena for showing off our requirements from the 2D env.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import json

import argparse
from shape_herd.gym_wrappers.twod_playground_env import (
    TwoDPlaygroundEnv,
    NormalizeActions,
)
import os


def main(args):
    dense_rew = True
    env = TwoDPlaygroundEnv(
        dense_reward=args.type == "dense",
        render_game=False,
        time_limit=args.time_limit,
        action_repeat=args.action_repeat,
    )
    dir_name = os.path.join(args.logdir, f"twod_{args.type}v6", args.id)
    os.makedirs(dir_name, exist_ok=True)
    print(f"Logging to {dir_name}")
    n_games = args.ngames
    ep_tot_r = []
    ep_reww = []
    scores = {f"twod_{args.type}": {"random": 0.0}}
    for i in tqdm(range(n_games)):
        done = False
        tot_r = 0.0
        step = 0.0
        reww = []
        obs = env.reset()
        obss = [obs]
        ep_len = args.time_limit // args.action_repeat
        if args.pbar:
            pbar = tqdm(total=args.time_limit)
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_r += reward
            assert type(reward) == float, f"{reward}, {type(reward)}, {step}"
            reww.append(reward)
            obss.append(obs)
            step += env.action_repeat
            if args.pbar:
                pbar.update(env.action_repeat)
        env.close()

        tqdm.write(f"Episode Reward: {tot_r}")
        tqdm.write(f"Episode length: {len(reww)}")

        reww = reww + [reww[-1]] * (ep_len - len(reww))
        ep_reww.append(reww[:ep_len])
        ep_reww_np = np.array(ep_reww)
        np.savez_compressed(
            os.path.join(dir_name, f"rew_dist.npz"), ep_reww_np=ep_reww_np
        )

        ep_tot_r.append(tot_r)
        scores[f"twod_{args.type}"]["random"] = np.mean(ep_tot_r)
        tqdm.write(f"Episodic Total Reward Mean: {np.mean(ep_tot_r)}")
        with open(os.path.join(dir_name, "scores.json"), "w") as fp:
            json.dump(scores, fp)

        plt.figure(figsize=(16, 9))
        ep_mean = ep_reww_np.mean(0)
        ep_std = ep_reww_np.std(0)
        plt.plot(np.arange(len(ep_mean)), ep_mean)
        plt.fill_between(
            np.arange(len(ep_mean)), ep_mean - ep_std, ep_mean + ep_std
        )
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Episodic Rew Distribution over Time")
        plt.grid()
        plt.savefig(os.path.join(dir_name, f"ep_env_rew.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--id", type=str, default="random")
    parser.add_argument("--type", type=str, default="dense")
    parser.add_argument("--time_limit", type=int, default=3000)
    parser.add_argument("--pbar", type=bool, default=False)
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--ngames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    sys.exit(main(args))
