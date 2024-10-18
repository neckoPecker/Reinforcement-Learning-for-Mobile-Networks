import gymnasium as gym
import mobile_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import set_random_seed

import matplotlib.pyplot as plt


# Main
if __name__ == "__main__":
     vec_env = make_vec_env("mobile-small-central-v0", n_envs=1)
     model = PPO.load("./logs/ppo/mobile-small-central-v0_20/mobile-small-central-v0.zip", vec_env)
     mean_reward, std_reward = evaluate_policy(
         model,
         vec_env,
         n_eval_episodes=10,
         deterministic=True
     )
     print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")

     obs = vec_env.reset()
     for i in range(100):
        done = False
        while not done:
             action, _states = model.predict(obs)
             obs, rewards, done, info = vec_env.step(action)
             vec_env.render("human")
             

        print("Episode: " + str(i))
        print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")
