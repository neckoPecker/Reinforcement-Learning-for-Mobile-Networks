# The package used
import gymnasium as gym

# We installed 'gymnasium[classic-control] in order to get CartPole-v1
# The second arguement for initializing the environment is the render mode;
# This generates visualization so we see the RL learner be being trained
env = gym.make('CartPole-v1', render_mode='human')

#
observation, info = env.reset()

reward_counter = 0

for _ in range (10000):
    action = env.action_space.sample() # agent policy that uses the observation
    observation, reward, terminated, truncated, info = env.step(action)

    # print("---------- Iteration " + str(_) + " ----------")
    # print(observation)
    # print(reward)
    # print(terminated)
    reward_counter += reward
    print(reward_counter)
    # print(truncated)
    # print(info)
    # print("----------------------------------")

    if terminated or truncated:
        observation, info = env.reset()

env.close()
