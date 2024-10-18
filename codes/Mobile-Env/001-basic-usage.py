# The following code example was taken from the example usage in the
# github page (https://github.com/stefanbschneider/mobile-env) and
# slightly modified

import gymnasium as gym
import mobile_env
from time import sleep

# https://stackoverflow.com/questions/52251582/matplotlib-has-no-attribute-cm-when-deploying-an-app
# For some reason, this is required

env = gym.make("mobile-large-central-v0", render_mode='human')
obs, info = env.reset()

done = False


while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(obs)
    sleep(10)
        
    if terminated or truncated:
        observation, info = env.reset()
        done = True

