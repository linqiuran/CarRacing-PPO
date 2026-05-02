import cv2
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from car_racing import car_ppo_agent
import os
import json
from datetime import datetime
from tqdm import tqdm


# ==================== 环境预处理（复用你的 CarRacing Wrapper） ====================
class CarRacingWrapper(gym.Wrapper):
    """图像预处理 + 帧跳过"""

    def __init__(self, env, skip_frames=4, grayscale=True, resize_shape=(84, 84)):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.grayscale = grayscale
        self.resize_shape = resize_shape

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
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


class FrameStackWrapper(gym.Wrapper):
    """帧堆叠包装器"""

    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = []

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            stacked_shape = (obs_shape[0], obs_shape[1], n_stack)
        else:
            stacked_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * n_stack)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=stacked_shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_stack
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        stacked = np.stack(self.frames, axis=-1)
        return stacked.astype(np.float32) / 255.0


# ==================== 测试函数（新增） ====================
def test_agent(agent, num_episodes=5):
    """
    评估智能体，返回平均奖励和平均步数
    """
    # 创建独立的测试环境（不渲染）
    test_env = make_env(render_mode="rgb_array")()

    # 创建可视化窗口的测试环境
    # test_env = make_env(render_mode="human")()

    total_rewards = []
    total_lengths = []

    for ep in range(num_episodes):
        state, _ = test_env.reset()
        state = np.transpose(state, (2, 0, 1))

        episode_reward = 0
        episode_steps = 0

        while True:
            action, _, _ = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if terminated or truncated:
                total_rewards.append(episode_reward)
                total_lengths.append(episode_steps)
                break

        # print(f'total_rewards:{total_rewards}')
        # print(f'total_lengths:{total_lengths}')

    test_env.close()

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_length = np.mean(total_lengths)

    return mean_reward, std_reward, mean_length


# ==================== 训练配置（保持不变） ====================
def make_env(render_mode="rgb_array"):
    def _init():
        env = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=True,
            domain_randomize=False,
            max_episode_steps=4000,
        )
        env = CarRacingWrapper(env, skip_frames=4)
        env = FrameStackWrapper(env, n_stack=4)
        return env

    return _init


# ==================== 主训练函数（修改部分） ====================
def train_car_racing(step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("=" * 60)
    print("CarRacing PPO 训练配置")
    print("=" * 60)

    train_timesteps = step
    rollout_steps = 512
    gamma = 0.99
    lam = 0.95
    lr = 3e-4
    n_epochs = 10
    minibatch_size = 32
    eval_freq = 20000
    n_eval_episodes = 3

    #  日志保存配置
    log_path = "./training_logs"
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "training_log_9.json")
    training_logs = []  # 存储所有日志记录

    env = make_env()()

    agent = car_ppo_agent(
        env,
        lr=lr,
        batch_size=rollout_steps,
        max_size=rollout_steps,
        gamma=gamma,
        lam=lam,
        ent_coef=0.01,
    )

    total_steps = agent.load_model('models/car_racing_ppo_self_final.pth')
    agent.ent_coef = 0.0001
    agent.lr = 1e-5
    lr = agent.lr
    num_timesteps = total_steps + train_timesteps

    # next_eval_step = eval_freq
    next_eval_step = ((total_steps // eval_freq) + 1) * eval_freq

    # target_std = 1.5
    # with torch.no_grad():
    #     # 假设你的动作空间是3维：[转向, 油门, 刹车]
    #      agent.policy.log_std.data = torch.tensor(
    #         [np.log(target_std), np.log(target_std), np.log(target_std)],
    #         device=agent.device,  # 使用模型所在的设备
    #         dtype=torch.float32
    #     )

    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    episode_reward = 0
    episode_steps = 0

    best_eval_reward = -float('inf')
    

    state, _ = env.reset()
    state = np.transpose(state, (2, 0, 1))

    # total_steps = 0
    iteration = 0

    print("\n开始训练...\n")
    pbar = tqdm(total=train_timesteps, desc="训练进度")

    print(f' 训练步数为:{train_timesteps}')
    print(f"  总步数: {num_timesteps:,}")
    print(f"  n_steps: {rollout_steps}")
    print(f"  batch_size: {minibatch_size}")
    print(f"  学习率: {lr}")
    print(f"  评估频率: {eval_freq} 步")
    print(f"  日志保存: {log_file}")
    print("=" * 60)

    while total_steps < num_timesteps:
        # ===== 收集经验 =====
        for _ in range(rollout_steps):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = np.transpose(next_state, (2, 0, 1))
            agent.buffer.push(state, reward, action, log_prob, done, next_state, value)

            episode_reward += reward
            episode_steps += 1
            state = next_state
            total_steps += 1

            pbar.update(1)

            if done:
                ep_rewards.append(episode_reward)
                ep_lengths.append(episode_steps)
                episode_reward = 0
                episode_steps = 0
                state, _ = env.reset()
                state = np.transpose(state, (2, 0, 1))

            if total_steps >= num_timesteps:
                break

        # ===== PPO 更新 =====
        iteration += 1
        pg_loss, v_loss, entropy, explained_var = agent.update()

        # ===== 获取训练指标 =====
        if len(ep_rewards) > 0:
            ep_rew_mean = np.mean(ep_rewards)
            ep_len_mean = np.mean(ep_lengths)
        else:
            ep_rew_mean = 0
            ep_len_mean = 0

        current_std = agent.policy.log_std.exp().mean().item()

        # ===== 打印训练日志 =====
        print("-" * 40)
        print(f"| rollout/                |            |")
        print(f"|    ep_len_mean          | {ep_len_mean:<.0f}        |")
        print(f"|    ep_rew_mean          | {ep_rew_mean:<.2f}       |")
        print(f"| time/                   |            |")
        print(f"|    iterations           | {iteration}         |")
        print(f"|    total_timesteps      | {total_steps}      |")
        print(f"| train/                  |            |")
        print(f"|    learning_rate        | {lr}      |")
        print(f"|    loss                 | {pg_loss + v_loss:.3f}       |")
        print(f"|    policy_gradient_loss | {pg_loss:.4f}     |")
        print(f"|    std                  | {current_std:.3f}        |")
        print(f"|    value_loss           | {v_loss:.4f}        |")
        print(f"|    explained_variance   | {explained_var:.3f}        |")
        print("-" * 40)

        # =====  新增：记录训练日志到 JSON =====
        log_entry = {
            "timesteps": total_steps,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "ep_rew_mean": float(ep_rew_mean),
            "ep_len_mean": float(ep_len_mean),
            "pg_loss": float(pg_loss),
            "v_loss": float(v_loss),
            "total_loss": float(pg_loss + v_loss),
            "entropy": float(entropy),
            "std": float(current_std),
            "explained_variance": float(explained_var)
        }
        training_logs.append(log_entry)

        # 每 10 次迭代保存一次日志（避免频繁写入）
        if iteration % 10 == 0:
            with open(log_file, 'w') as f:
                json.dump(training_logs, f, indent=2)

        # ===== 每 20000 步评估一次 =====
        if total_steps >= next_eval_step:
            eval_reward, eval_std, eval_length = test_agent(agent, num_episodes=n_eval_episodes)

            print("\n" + "=" * 40)
            print(f"Eval num_timesteps={total_steps}, episode_reward={eval_reward:.2f} +/- {eval_std:.2f}")
            print(f"Episode length: {eval_length:.2f} +/- 0.00")
            print("-" * 40)
            print(f"| eval/                   |            |")
            print(f"|    mean_ep_length       | {eval_length:<.0f}        |")
            print(f"|    mean_reward          | {eval_reward:.2f}       |")
            print(f"| time/                   |            |")
            print(f"|    total_timesteps      | {total_steps}      |")
            print("-" * 40 + "\n")

            #  新增：评估日志记录
            eval_entry = {
                "type": "eval",
                "timesteps": total_steps,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "eval_reward": float(eval_reward),
                "eval_std": float(eval_std),
                "eval_length": float(eval_length),
            }
            training_logs.append(eval_entry)

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_model(f"models/best_model_self_ppo_{total_steps}.pth", total_steps)
                print(f"New best mean reward! 模型已保存\n")

            next_eval_step += eval_freq

    # ===== 保存最终日志 =====
    with open(log_file, 'w') as f:
        json.dump(training_logs, f, indent=2)

    # 保存最终模型
    agent.save_model("models/car_racing_ppo_self_final.pth", total_steps)
    print(f"\n训练完成！日志已保存到 {log_file}")

    return agent


if __name__ == "__main__":
    # 模型训练
    # step = 500000
    # train_car_racing(step)

    # 测试模型
    env = make_env()()
    agent = car_ppo_agent(
        env,
    )

    total_steps = agent.load_model('models/best_model_self_ppo_2940032.pth')
    agent.ent_coef = 0.0001
    agent.lr = 1e-5
    mean_reward, std_reward, mean_length = test_agent(agent, 10)
    print(f'mean_reward:{mean_reward},std_reward:{std_reward},:mean_length:{mean_length}')
  
