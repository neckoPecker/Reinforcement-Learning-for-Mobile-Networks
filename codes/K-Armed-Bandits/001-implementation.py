import numpy as np
import random
import matplotlib.pyplot as plt

# Distribution was taken from textbook
distribution = {
    1: (-2, 2),
    2: (-2, 1),
    3: (-1, 3),
    4: (-2, 2),
    5: (-1, 3),
    6: (-3, 1),
    7: (-3, 2),
    8: (-3, 1),
    9: (-1, 3),
    10: (-3, 1),
}

Q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Initial distribution value is 0
rewards = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
actions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

reward_summary = []

epsilon = 0.1;
for episode in range (100):

    print("EPISODE " + str(episode))
    for step in range (10):
        
        print("\tSTEP " + str(step))
        
        # Exploration; Choose a random option
        if random.random() > (1 - epsilon):
            choice = random.randint(1, 10);
            reward = random.randint(distribution[choice][0], distribution[choice][1])
            obtained_reward = rewards[-1]
            obtained_reward[choice-1] += reward
            print("\t\t" + str(obtained_reward) + "\t(Exploration)" + "\tIs Loss? " + str(reward < 0))
            rewards = np.vstack([rewards, obtained_reward])
            actions[choice-1] += 1 

        # Exploitation: Choose the greedy option (arbitrary if there are ties)
        else:
            choice = (rewards.sum(axis=0) / actions).argmax() + 1
            reward = random.randint(distribution[choice][0], distribution[choice][1])
            obtained_reward = rewards[-1]
            obtained_reward[choice-1] += reward
            print("\t\t" + str(obtained_reward) + "\t(Greedy)" + "\tIs Loss? " + str(reward < 0))
            rewards = np.vstack([rewards, obtained_reward])
            actions[choice-1] += 1
    reward_summary.append(rewards.sum().sum())
    # rewards.fill(0)
    

print(reward_summary)
plt.plot(reward_summary)
plt.show()
