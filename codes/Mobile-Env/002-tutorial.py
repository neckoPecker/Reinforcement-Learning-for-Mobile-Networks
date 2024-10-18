# The following code was taken from
# https://github.com/stefanbschneider/mobile-env, which in turn lead
# to this website:
# https://colab.research.google.com/github/stefanbschneider/mobile-env/blob/master/examples/demo.ipynb

import gymnasium as gym
import matplotlib.pyplot as plt
import mobile_env

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

env = gym.make("mobile-medium-central-v0", render_mode="human")
# env = CustomEnv(config={'handler': CustomHandler})
model = PPO(MlpPolicy, env, tensorboard_log='results_sb', verbose=1)
model.learn(total_timesteps=300)
print(f"\nEnvironment with {env.unwrapped.NUM_USERS} users and {env.unwrapped.NUM_STATIONS} cells.")

done = False
obs, info = env.reset()

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Render the scenario
    env.render()

    if terminated or truncated:
        observation, info = env.reset()
        done = True

env.close()
