import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

vec_env = make_vec_env("CartPole-v1", n_envs=4)
# model = PPO("MlpPolicy", vec_env, verbose=1)
model = PPO.load("./ppo_cartpole.zip/", vec_env)

mean_reward, std_reward = evaluate_policy(
    model,
    vec_env,
    n_eval_episodes=10,
    deterministic=True
)

# model.learn(total_timesteps=int(1e5), progress_bar=True)
# model.save("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")
    # print(f"Obs: {obs}\tRew: {rewards}\tDone?: {dones}\tInfo: {info}")
