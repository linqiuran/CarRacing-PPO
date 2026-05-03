# CarRacing PPO 深度调优

##  项目简介
使用 PPO 算法在 CarRacing-v3 环境中训练自动驾驶策略，解决了稀疏奖励环境下的训练崩溃与探索失效问题。

##  核心挑战与解决方案

### 1. 策略崩溃（Reward 从 +173 暴跌到 -392）
- **诊断**：`explained_variance` 降至 -10，判定为价值网络崩溃
- **修复**：回滚到崩溃前权重，引入 `clip_range_vf` + `target_kl` 约束

### 2. 确定性策略失效（测试时车辆不动）
- **诊断**：`std` 异常增大至 8.8，策略完全依赖噪声
- **修复**：手动重置 `log_std` + 分阶段熵衰减

##  实验数据
| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| Eval Reward | -398 | +194 |
| Action Std | 8.8 | 1.1 |
| Explained Variance | -10 | 0.8 |

##  快速运行
\`\`\`bash
# 1. 安装依赖
pip install -r requirements.txt


##  技术栈
- Python, PyTorch, Stable-Baselines3
- Gymnasium, OpenCV, NumPy
