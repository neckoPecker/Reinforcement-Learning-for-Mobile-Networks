# Followed Aleksandar Haber's Tutorial for Cart Pole problem.
# https://aleksandarhaber.com/cart-pole-control-environment-in-openai-gym-gymnasium-introduction-to-openai-gym/

# For saving feature, I used this
# https://github.com/openai/gym/issues/402

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import pickle

env = gym.make('CartPole-v1')

"""

env.reset()

	Returns 2 variables: observation (ObsType) and info (dictionary). The first is an observation of
	the initial state. The second is a dictionary that contains additional information regarding the
	observation

	Here, we store these two values in the variables 'state' and "_".

	Source: https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
"""
(state,_) = env.reset()

num_of_episodes = 1000000
steps_per_episode = 500

final_rewards = []

for episode_index in range(num_of_episodes):
    initial_state = env.reset()
    print("EPISODE " + str(episode_index))
    # env.render()

    for step_index in range (steps_per_episode):
        agent_action = env.action_space.sample()
        (observation, reward, terminated, truncated, info) = env.step(agent_action)
        output = "\t{:<20s} {:<20s} {:<20s}".format("Current Step: " + str(step_index), "Reward: " + str(reward), "Terminated? " + str(terminated))
        print(output)
        # time.sleep(0.1)
        if (terminated):
            # time.sleep(1)
            final_rewards.append(step_index)
            break
env.close()

# Visualize the results
plt.plot(final_rewards)
plt.show()

# Save the environment (well, at least just this one)
# save_point_env = copy.deepcopy(env)
# with open("test.pkl", "wb") as pkl_file:
#     pickle.dump(save_point_env, pkl_file)
