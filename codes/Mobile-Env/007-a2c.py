import gymnasium as gym
import mobile_env

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

def make_subproc_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

# Main 
if __name__ == "__main__":

    # Make the environment
    # env = SubprocVecEnv([make_subproc_env("mobile-small-central-v0", i) for i in range(8)])
    env = make_vec_env("mobile-small-central-v0", n_envs=1)

    # Make, Train, Save
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./007-a2c-tensorboard/")
    model = A2C.load("./007-model.zip")
    # model.learn(total_timesteps=int(1e6), progress_bar=True)
    # model.save("007-model")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean rewards {mean_reward}\tStd Dev: {std_reward}")

    # See how it performs
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render("human")
