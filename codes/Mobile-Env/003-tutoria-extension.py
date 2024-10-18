"""#003-tutoria-extension.py

In this file, we attempt to learn how to extend mobile-env so that we
are more aware of its flexibility. This may allow for us to customize
it for our purposes, as well as understand more about how mobile-env
works behind the hood.

"""
import numpy as np
import gymnasium
import mobile_env

from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment

# overall number of active connections
def overall_connections(sim):
    return sum([len(conns) for conns in sim.connections.values()])


# monitors utility per user equipment
def user_utility(sim):
    return {ue.ue_id: utility for ue, utility in sim.utilities.items()}


# monitors each user equipments' distance to their closest base station
def user_closest_distance(sim):
    # position vector of basestations
    bpos = np.array([[bs.x, bs.y] for bs in sim.stations.values()])

    distances = {}    
    for ue_id, ue in sim.users.items():
        upos = np.array([[ue.x, ue.y]])
        dist = np.sqrt(np.sum((bpos - upos)**2, axis=1)).min()
        
        distances[ue_id] = dist
    
    return distances


# number of connections per basestation
def station_connections(sim):
    return {bs.bs_id: len(conns) for bs, conns in sim.connections.items()}

# add custom metrics to config of environment
config = {"metrics": {
            "scalar_metrics": {"overall connections": overall_connections},
            "ue_metrics": {"user utility": user_utility, 'distance station': user_closest_distance},
            "bs_metrics": {"station connections": station_connections}
            }
         }

class CustomEnv(MComCore):
    # overwrite the default config
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            # 10 steps per episode
            "EP_MAX_TIME": 10,
            # identical episodes
            "seed": 1234,
            'reset_rng_episode': True,
        })
        # faster user movement
        config["ue"].update({
            "velocity": 10,
        })
        return config

    # configure users and cells in the constructor
    def __init__(self, config={}, render_mode=None):
        # load default config defined above; overwrite with custom params
        env_config = self.default_config()
        env_config.update(config)

        # two cells next to each other; unpack config defaults for other params
        stations = [
            BaseStation(bs_id=0, pos=(50, 100), **env_config["bs"]),
            BaseStation(bs_id=1, pos=(100, 100), **env_config["bs"])
        ]

        # users
        users = [
            # two fast moving users with config defaults
            UserEquipment(ue_id=1, **env_config["ue"]),
            UserEquipment(ue_id=2, **env_config["ue"]),
            # stationary user --> set velocity to 0
            UserEquipment(ue_id=3, velocity=0, snr_tr=env_config["ue"]["snr_tr"], noise=env_config["ue"]["noise"],
                          height=env_config["ue"]["height"]),
        ]

        super().__init__(stations, users, config, render_mode)


# env = gymnasium.make("mobile-small-central-v0", config=config, render_mode="human")
env = CustomEnv(render_mode="human", config=config)
print(f"\nSmall environment with {env.NUM_USERS} users and {env.NUM_STATIONS} cells.")

# The loop
done = False
obs, info = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    done = terminated or truncated

    

scalar_results, ue_results, bs_results = env.monitor.load_results()

print("\n\n\n")
print(scalar_results.head())
