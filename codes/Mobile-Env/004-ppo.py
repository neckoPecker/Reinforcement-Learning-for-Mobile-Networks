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

def make_env(env_id: str, rank: int, seed: int = 0):

    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

class RenderCallback(BaseCallback):

    def __init__(self, vec_env: VecEnv, verbose: int = 0, render: bool = False):
        super().__init__(verbose)
        self.model = model
        self.render = render
        print(model.action_space)

    def _on_step(self) -> bool:
        # Rendering
        if self.render:
            vec_env.render("human")        
        return True

class TensorboardCallback(BaseCallback):

    def __init__(self, vec_env: VecEnv, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        return True

# Main
if __name__ == "__main__":
     vec_env = make_vec_env("mobile-small-central-v0", n_envs=1)
     # vec_env = SubprocVecEnv([make_env("mobile-small-central-v0", i) for i in range(4)])
     
     # custom_objects = {
     #     'learning_rate': 0.5    
     # }

     # model = PPO.load("./ppo_small_mobile_env.zip", vec_env)
     model = PPO.load("./archive/logs/ppo/mobile-small-central-v0_21/mobile-small-central-v0.zip", vec_env)
     # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./004-ppo-tensorboard/")

     # model.learn(total_timesteps=int(10 * 1e6), progress_bar=True)
     # model.save("ppo_small_mobile_env")

     mean_reward, std_reward = evaluate_policy(
         model,
         vec_env,
         n_eval_episodes=10,
         deterministic=True
     )
     print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")

     
     obs = vec_env.reset()

     checkpoint_callback = CheckpointCallback(
         save_freq=100,
         save_path="./004-ppo-logs/",
         name_prefix="rl_model",
         save_replay_buffer=True,
         save_vecnormalize=True)

     render_callback = RenderCallback(vec_env, False)
     tb_callback = TensorboardCallback(vec_env)
     callbacks = CallbackList([render_callback,
                               checkpoint_callback,
                               tb_callback])
     # NOTE: tb_callback doesn't do anything
     # model.learn(total_timesteps=int(1e5), callback=callbacks, progress_bar=True, tb_log_name="run")
     # model.save("ppo_small_mobile_env")


     for i in range(100):
        done = False
        while not done:
             action, _states = model.predict(obs)
             obs, rewards, done, info = vec_env.step(action)
             vec_env.render("human")
             

        print("Episode: " + str(i))
        print(f"Mean reward: {mean_reward}\tstd Dev: {std_reward}")
        
