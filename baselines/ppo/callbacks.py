import os
import io
import cv2
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image as loggerImage
from stable_baselines3.common.logger import Video as loggerVideo
import torch


def fig2data(fig, dpi=72):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @param dpi DPI of saved image
    @return a numpy 3D array of RGBA values
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class SaveVideoCallback(BaseCallback):
    """
    Callback for saving the setpoint tracking plot(the check is done every ``eval_freq`` steps)

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(
        self, eval_env, eval_freq=10000, log_dir=None, nenvs=10, verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = None
        self.nenvs = nenvs
        if log_dir is not None:
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, "images")

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            ep_tot_r = []
            for i in range(self.nenvs):
                obs = self.eval_env.reset()
                obss = [obs]
                done = False
                tot_r = 0.0
                reww = []
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    obss.append(obs)
                    reww.append(reward)
                    tot_r += reward
                torch_obss = (
                    torch.from_numpy(np.array(obss))
                    .unsqueeze(0)
                    .permute(0, 1, 4, 2, 3)
                )
                ep_tot_r.append(tot_r)
            ep_tot_r_mean = np.mean(ep_tot_r)
            print("Evaluation Reward: ", ep_tot_r_mean)
            print(torch_obss.shape)
            self.logger.record("scalars/eval_return", ep_tot_r_mean)
            if self.n_calls % 10 * (self.eval_freq) == 0:
                self.logger.record(
                    "eval_policy",
                    loggerVideo(torch_obss, 30),
                    exclude=("stdout", "log", "json", "csv"),
                )
            if self.save_path is not None:
                clip = ImageSequenceClip(list(obss), fps=20)
                video_dir = os.path.join(self.log_dir, "gifs")
                os.makedirs(video_dir, exist_ok=True)
                clip.write_gif(
                    os.path.join(video_dir, "eval_policy.gif"),
                    fps=20,
                    program="ffmpeg",
                    verbose=False,
                    logger=None,
                )

            plt.figure(figsize=(16, 4))
            lineObj = plt.plot(np.arange(len(reww)), reww)
            plt.ylabel("Reward")
            plt.xlabel("Steps")
            plt.xlim(0, len(reww))
            plt.grid()
            plt.title("Reward")

            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            # if self.n_calls % (50 * self.eval_freq) == 0:
            #     self.logger.record("Reward", loggerImage(img, "HWC"), exclude=("stdout", "log", "json", "csv"))
            if self.save_path is not None:
                im = Image.fromarray(img)
                path = os.path.join(self.save_path, "rews.png")
                im.save(path)

        return True
