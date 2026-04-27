import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import os
import json
from datetime import datetime

# ==================== 简单的日志回调（替代 TensorBoard） ====================
class SimpleLogCallback(BaseCallback):
    """简单的训练日志回调，不依赖 TensorBoard"""
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = os.path.join(log_path, "training_log.json")
        self.logs = []
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            # 每 1000 步记录一次
            log_data = {
                "timesteps": self.num_timesteps,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                
                log_data["mean_reward"] = np.mean(rewards) if rewards else 0
                log_data["mean_length"] = np.mean(lengths) if lengths else 0
                log_data["num_episodes"] = len(rewards)
                
                # 打印到控制台
                print(f"Step: {self.num_timesteps:>8} | "
                      f"Mean Reward: {log_data['mean_reward']:>7.2f} | "
                      f"Mean Length: {log_data['mean_length']:>6.0f}")
            
            self.logs.append(log_data)
            
            # 保存日志到文件
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        
        return True


# ==================== 环境预处理包装器 ====================
class CarRacingWrapper(gym.Wrapper):
    """
    针对 CarRacing-v3 的预处理包装器
    - 将 96x96 RGB 转为 84x84 灰度图
    - 帧跳过（动作重复）
    - 奖励裁剪（稳定训练）
    """
    def __init__(self, env, skip_frames=4, grayscale=True, resize_shape=(84, 84)):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.grayscale = grayscale
        self.resize_shape = resize_shape
        
        # 更新观察空间
        if grayscale:
            obs_shape = resize_shape
        else:
            obs_shape = (*resize_shape, 3)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess(obs)
        return obs, info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip_frames):
            obs, reward, term, trunc, step_info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            info.update(step_info)
            if terminated or truncated:
                break
                
        obs = self._preprocess(obs)
        total_reward = np.clip(total_reward, -1.0, 1.0)
        return obs, total_reward, terminated, truncated, info
    
    def _preprocess(self, obs):
        """图像预处理：RGB -> 灰度 -> 缩放"""
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


class FrameStackWrapper(gym.Wrapper):
    """
    帧堆叠包装器
    将连续多帧堆叠成通道维度
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = []
        
        # 获取原始观察空间的形状
        obs_shape = env.observation_space.shape
        
        # 创建堆叠后的观察空间
        # 如果是灰度图 (H, W)，堆叠后变成 (H, W, n_stack)
        if len(obs_shape) == 2:
            stacked_shape = (obs_shape[0], obs_shape[1], n_stack)
        else:
            stacked_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * n_stack)
            
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=stacked_shape, dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 初始时用第一帧填充所有堆叠帧
        self.frames = [obs] * self.n_stack
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        # 将帧堆叠在最后一个维度并归一化到 [0, 1]
        stacked = np.stack(self.frames, axis=-1)
        return stacked.astype(np.float32) / 255.0


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    将观察从 (H, W, C) 转换为 (C, H, W) 以适应 PyTorch CNN
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        # 转换为 (C, H, W) 格式
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


def make_env(render_mode="rgb_array", continuous=True, skip_frames=4, n_stack=4):
    """创建并包装环境"""
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=continuous,
            domain_randomize=False,
            max_episode_steps = 4000,
        )
        env = Monitor(env)
        env = CarRacingWrapper(env, skip_frames=skip_frames)
        env = FrameStackWrapper(env, n_stack=n_stack)
        env = ChannelFirstWrapper(env)
        
        return env
    return _init


# ==================== 极小 CNN 特征提取器 ====================
class MinimalCarRacingCNN(BaseFeaturesExtractor):
    """
    最小显存占用的 CNN 特征提取器
    通道数: [8, 16, 16]
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算卷积输出尺寸
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ==================== 改进版继续训练 ====================
def improved_continue_training(model_path="./car_racing_continued/final_2450768.zip"):
    """
    改进的继续训练，包含更好的配置
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("继续训练配置")
    print("=" * 60)
    
    # 加载模型
    model = PPO.load(
        model_path,
        device=device,
        custom_objects={
            "policy_kwargs": dict(
                features_extractor_class=MinimalCarRacingCNN,
                features_extractor_kwargs=dict(features_dim=128),
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            )
        }
    )
    
    current_timesteps = model.num_timesteps
    print(f"当前训练步数: {current_timesteps:,}")
    
    # 创建环境
    train_env = DummyVecEnv([make_env(render_mode="rgb_array")])
    model.set_env(train_env)
    
    eval_env = DummyVecEnv([make_env(render_mode="rgb_array")])
    
    # 调整学习率（随着训练进行，降低学习率）
    model.learning_rate = 1e-5  # 从 3e-4 降低到 1e-4
    
    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./car_racing_continued",
        name_prefix="continued_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./car_racing_continued/best_model",
        eval_freq=20000,
        n_eval_episodes=3,
        deterministic=True,
    )
    
    # 继续训练 100 万步
    additional_timesteps = 500_000
    target_timesteps = current_timesteps + additional_timesteps
    
    print(f"目标总步数: {target_timesteps:,}")
    print(f"学习率: {model.learning_rate}")
    print("=" * 60)

    # target_std = 1.5
    # with torch.no_grad():
    #     # 假设你的动作空间是3维：[转向, 油门, 刹车]
    #      model.policy.log_std.data = torch.tensor(
    #         [np.log(target_std), np.log(target_std), np.log(target_std)],
    #         device=model.device,  # 使用模型所在的设备
    #         dtype=torch.float32 
    #     )

    model.ent_coef = 0.0001

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            reset_num_timesteps=False,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断，正在保存模型...")
        model.save(f"./car_racing_ppo_minimal/final_model_minimal.zip")
    
    model.save(f"./car_racing_continued/final_{target_timesteps}.zip")
    print(f"训练完成！总步数: {model.num_timesteps:,}")


# ==================== 激进显存优化配置 ====================
def train_car_racing_minimal(
    total_timesteps=1_000_000,
    save_path="./car_racing_ppo_minimal",
    log_path="./car_racing_logs",
    use_gpu=True,
):
    """
    最小显存占用配置（适合 4-8GB 显存）
    不依赖 TensorBoard
    """
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # 设备配置
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 🔧 单环境训练
    n_envs = 1
    print(f"正在创建 {n_envs} 个训练环境...")
    train_env = DummyVecEnv([make_env(render_mode="rgb_array") for _ in range(n_envs)])
    
    print("正在创建评估环境...")
    eval_env = DummyVecEnv([make_env(render_mode="rgb_array")])
    
    # 回调函数 - 不使用 TensorBoard
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=log_path,
        eval_freq=20000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="car_racing_model"
    )
    
    # 简单的日志回调
    simple_log_callback = SimpleLogCallback(log_path)
    
    # 策略网络配置
    policy_kwargs = dict(
        features_extractor_class=MinimalCarRacingCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    
    print("正在创建 PPO 模型...")
    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=32,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        # 注意：这里不设置 tensorboard_log
        device=device,
    )
    
    print(f"\n{'='*60}")
    print(f"极小显存配置：")
    print(f"  - 并行环境数: {n_envs}")
    print(f"  - n_steps: 512")
    print(f"  - batch_size: 32")
    print(f"  - CNN 通道数: [8, 16, 16]")
    print(f"  - 全连接层: [64, 64]")
    print(f"  - 总训练步数: {total_timesteps:,}")
    print(f"  - 日志保存: {log_path}/training_log.json")
    print(f"{'='*60}\n")
    
    # 开始训练
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, simple_log_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断，正在保存模型...")
    
    # 保存最终模型
    final_model_path = f"{save_path}/final_model_minimal.zip"
    model.save(final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")
    
    # 清理环境
    train_env.close()
    eval_env.close()
    
    return model


# ==================== 测试模型 ====================
def test_model(model_path, num_episodes=3, render=True):
    """
    加载并测试训练好的模型
    """
    render_mode = "human" if render else "rgb_array"
    
    # 创建测试环境
    env = make_env(render_mode=render_mode)()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_path}")
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        model = PPO.load(model_path, device=device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用自定义 CNN 加载...")
        # 如果直接加载失败，使用自定义参数加载
        model = PPO.load(
            model_path, 
            device=device,
            custom_objects={
                "policy_kwargs": dict(
                    features_extractor_class=MinimalCarRacingCNN,
                    features_extractor_kwargs=dict(features_dim=128),
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                )
            }
        )
    
    print(f"\n开始测试，共 {num_episodes} 个 episode")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if info.get("lap_completed"):
                print(f"✅ 成功打卡完一圈！得分: {total_reward}")

            if terminated or truncated:
                print(f"Episode {episode + 1}: 奖励 = {total_reward:.2f}, 步数 = {steps}")
                break

        
    
    env.close()
    print("\n测试完成！")


# ==================== 监控显存使用 ====================
def print_gpu_memory():
    """打印当前 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 显存: 已分配 {allocated:.2f} GB / 缓存 {cached:.2f} GB / 总量 {total:.2f} GB")
    else:
        print("CUDA 不可用")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="CarRacing-v3 PPO 训练脚本 (8GB显存优化版，无TensorBoard依赖)")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "test", "memory", "test_env"],
                        help="运行模式: train(训练), test(测试), memory(查看显存), test_env(测试环境)")
    parser.add_argument("--model_path", type=str, default="./car_racing_ppo_minimal/final_model_minimal.zip",
                        help="测试时使用的模型路径")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="训练总步数（默认200万）")
    parser.add_argument("--no_gpu", action="store_true",
                        help="禁用 GPU，强制使用 CPU")
    parser.add_argument("--episodes", type=int, default=3,
                        help="测试时运行的 episode 数量")
    parser.add_argument("--save_path", type=str, default="./car_racing_ppo_minimal",
                        help="模型保存路径")
    parser.add_argument("--log_path", type=str, default="./car_racing_logs",
                        help="日志保存路径")
    
    args = parser.parse_args()

    

    model_path = "car_racing_continued/best_model/best_model.zip"
    
    test_model(model_path, num_episodes=args.episodes)
    
    # if args.mode == "memory":
    #     # 查看显存使用
    #     print_gpu_memory()
        
    # elif args.mode == "test_env":
    #     # 测试环境
    #     test_environment()
    #     print_gpu_memory()
        
    # elif args.mode == "train":
    #     # 先测试环境
    #     test_environment()
        
    #     # 训练前查看显存
    #     print_gpu_memory()
        
    #     #继续训练
    #     print("继续训练")
    #     model = improved_continue_training()
        

        #开始训练
        # model = train_car_racing_minimal(
        #     total_timesteps=args.timesteps,
        #     save_path=args.save_path,
        #     log_path=args.log_path,
        #     use_gpu=not args.no_gpu,
        # )
        
        # # 训练后查看显存
        # print_gpu_memory()
        
        # print("\n" + "=" * 60)
        # print("训练完成！")
        # print(f"训练日志保存在: {args.log_path}/training_log.json")
        # print("=" * 60)
        
    # elif args.mode == "test":
    #     # 测试模型
    #     if not os.path.exists(args.model_path):
    #         print(f"错误：模型文件不存在 {args.model_path}")
    #         print("请先训练模型或指定正确的模型路径")
    #     else:
    #         test_model(args.model_path, num_episodes=args.episodes)