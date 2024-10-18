""".#001-example.py

See https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf for
what the value function calculations represent

"""

import numpy as np


# State Transition Probabilities
t_probs = np.array(([0, 0.5, 0, 0, 0, 0.5, 0],	 # Class 1
                    [0, 0, 0.8, 0, 0, 0, 0.2], 	 # Class 2
                    [0, 0, 0, 0.6, 0.4, 0, 0],   # Class 3
                    [0, 0, 0, 0, 0, 0, 1.0],     # Pass
                    [0.2, 0.4, 0.4, 0, 0, 0, 0], # Pub
                    [0.1, 0, 0, 0, 0, 0.9, 0],   # Facebook
                    [0, 0, 0, 0, 0, 0, 1.0]))    # Sleep

# Reward Column
rewards = np.array(([-2],       # Class 1
                    [-2],       # Class 2
                    [-2],       # Class 3
                    [+10],      # Pass
                    [+1],       # Pub
                    [-1],       # Facebook
                    [0],        # Sleep
                    ))

# Discount factor (between 0 and 1)
gamma = 0.9

v = np.dot(np.linalg.inv(np.identity(len(t_probs)) - (gamma * t_probs)), rewards)
print(v)
