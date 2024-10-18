import gymnasium as gym
from pettingzoo import ParallelEnv
import mobile_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

env = ParallelEnv("mobile-small-ma-v0", render_mode="human")

obs, info = env.reset()
done = False
while not done:
    # action, = env.action_space[0].sample()
    action.env
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
