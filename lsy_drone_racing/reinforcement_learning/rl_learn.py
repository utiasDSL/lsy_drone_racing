import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn
from datetime import datetime
import os
from pathlib import Path

from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from docs import conf
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.envs.drone_race import DroneRaceEnv
from rl_env import RLDroneRacingWrapper, RenderCallback

# === 1. 创建训练环境 ===
config = load_config(Path(__file__).parents[2] / "config" / "levelrl.toml")
def make_env():
    base_env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    base_env = JaxToNumpy(base_env)
    wrapped_env = RLDroneRacingWrapper(base_env)
    return wrapped_env

env = DummyVecEnv([make_env])  # SB3 要求使用 VecEnv（即使只有1个）

# === 2. 设置训练保存目录和回调 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path(__file__).parent / f"ppo_logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_checkpoint")
eval_callback = EvalCallback(env, best_model_save_path=log_dir, eval_freq=5000, n_eval_episodes=5)

# === 3. 初始化 PPO 模型 ===
policy_kwargs = dict(
    net_arch=[128, 128],         # 两层，每层 128
    activation_fn=nn.ReLU        # 激活函数（默认是 Tanh，可以改为 ReLU）
)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    learning_rate=3e-4,
    ent_coef=0.0,
)

# === 4. 启动训练 ===
render_callback = RenderCallback(render_freq=100)
model.learn(total_timesteps=200000, callback=[checkpoint_callback, eval_callback, render_callback])

# === 5. 保存最终模型 ===
model.save(f"{log_dir}/ppo_final_model")
print(f"✅ 训练完成，模型已保存至 {log_dir}")
