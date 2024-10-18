import gymnasium as gym
import mobile_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):

    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

# Main
if __name__ == "__main__":
     vec_env = make_vec_env("mobile-large-central-v0", n_envs=1)
     # vec_env = SubprocVecEnv([make_env("mobile-large-central-v0", i) for i in range(16)])
     
     custom_objects = {
         'learning_rate': 0.5    
     }

     # model = PPO.load("./ppo_large_mobile_env.zip", vec_env)

     model = PPO("MlpPolicy", vec_env, verbose=1)

     # model.learn(total_timesteps=int(1e6), progress_bar=True)
     # model.save("ppo_small_mobile_env")

     mean_reward, std_reward = evaluate_policy(
         model,
         vec_env,
         n_eval_episodes=10,
         deterministic=True
     )
     print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")

     obs = vec_env.reset()
     while True:
         action, _states = model.predict(obs)
         obs, rewards, dones, info = vec_env.step(action)
         vec_env.render("human")
         # print(f"Obs: {obs}\tRew: {rewards}\tDone?: {dones}\tInfo: {info}")
