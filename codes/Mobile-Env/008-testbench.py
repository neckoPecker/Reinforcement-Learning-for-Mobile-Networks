import gymnasium as gym
import mobile_env
import numpy as np
import optuna

from typing import Optional, Type, Union

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import TensorBoardOutputFormat

# https://stackoverflow.com/questions/69181347/stable-baselines3-log-rewards
class SummaryWriterCallback(BaseCallback):
    '''
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    '''

    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals['my_custom_info_dict']['my_custom_reward']
            for i in range(self.locals['env'].num_envs):
                self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i+1),
                                                     rewards[i],
                                                     self.n_calls)

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

# For hiding annoying warnings
# From: https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_vec_env(env_id: str, rank: int, seed: int = 0):

    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":
     vec_env = VecMonitor(SubprocVecEnv([make_vec_env("mobile-small-central-v0", i) for i in range(32)]))

     # model = PPO.load("./ppo_large_mobile_env.zip", vec_env)

     # Initialization
     model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./temp_008_tensoboard_log/",
                 # clip_range_vf=None,
                 # clip_range=0.2,
                 # ent_coef=0.0,
                 # batch_size=16,
                 # gae_lambda=0.0,
                 # n_steps=4096,
                 # learning_rate=0.0003,
                 # n_epochs=10,
                 # max_grad_norm=0.5,
                 # vf_coef=0.5,
                 # normalize_advantage=True
                 )

     # Learn and save
     model.learn(1e5, progress_bar=True, callback=SummaryWriterCallback())
     model.save("./temp_008_model")

     mean_reward, std_reward = evaluate_policy(
         model,
         vec_env,
         n_eval_episodes=10,
         deterministic=True
     )
     print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")

     # obs = vec_env.reset()
     # while True:
     #     action, _states = model.predict(obs)
     #     obs, rewards, dones, info = vec_env.step(action)
     #     vec_env.render("human")

