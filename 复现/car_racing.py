import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random


class ActorCritic(nn.Module):
    def __init__(self,action_dim,hidden=128):
        super(ActorCritic,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4,8,8,4),
            nn.ReLU(),
            nn.Conv2d(8,16,4,2),
            nn.ReLU(),
            nn.Conv2d(16,16,3,1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1,4,84,84)
            self.cnn_dim = self.cnn(sample).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(self.cnn_dim, hidden),
            nn.ReLU(),
            nn.Linear(128,action_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(self.cnn_dim,hidden),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):
        features = self.cnn(x)
        mean = self.actor(features)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(features)

        return mean, std, value

    def get_action(self,x,deterministic=False):
        mean,std,value = self.forward(x)
        dist = Normal(mean,std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)

class rebuff:
    def __init__(self,max):
        self.batch = deque(maxlen=max)

    def push(self,state,reward,action,log_prob,done,next_state,value):
        self.batch.append((state,reward,action,log_prob,done,next_state,value))

    def get_batch(self):
        states, rewards,actions, log_probs, dones, next_states, values = zip(*self.batch)

        return (np.stack(states),
                np.array(rewards, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(log_probs, dtype=np.float32),
                np.array(dones, dtype=np.bool_),
                np.stack(next_states),
                np.array(values, dtype=np.float32))

    def clear(self):
        self.batch.clear()

    def __len__(self):
        return len(self.batch)



class car_ppo_agent:
    def __init__(self,env,lr=3e-4,batch_size=512,max_size=512,gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.clip_epsilon=clip_epsilon
        self.ent_coef=ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr


        self.policy = ActorCritic(action_dim=3).to(self.device)
        self.optim = optim.Adam(self.policy.parameters(),lr=self.lr)

        self.buffer = rebuff(max_size)

    def get_action(self,state,deterministic=False):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        mean, std, value = self.policy(state_tensor)

        dist = Normal(mean,std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.squeeze(0).detach().cpu().numpy()
        log_prob_np = log_prob.squeeze(0).detach().cpu().item()
        value_np = value.squeeze(0).detach().cpu().item()

        return action_np, log_prob_np, value_np

    def compute_gae(self,values,rewards,dones):
        T = len(rewards)
        advantages = np.zeros(T,dtype=np.float32)
        last_advantages = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_advantages = delta + self.gamma * self.lam * (1-dones[t]) * last_advantages
            advantages[t] = last_advantages

        returns = advantages + values[:T]
        return advantages,returns

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0,0.0,0.0

        states,rewards,actions,old_log_probs,dones,next_states,values = self.buffer.get_batch()
        T = len(states)

        advantages,returns = self.compute_gae(values,rewards,dones)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            value_var = torch.var(returns_tensor)
            unexplained_var = torch.var(returns_tensor - values_tensor)
            explained_var = 1 - (unexplained_var / (value_var + 1e-8))

        states_batch = torch.tensor(states,dtype=torch.float32,device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean())  / (advantages.std() + 1e-8)


        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0

        for epoch in range(10):
            indices = torch.randperm(T)

            for i in range(0,T,64):
                batch_indices = indices[i:i + 64]

                b_states = states_batch[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                mean, std, values_pred = self.policy(b_states)
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)

                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_log_probs - b_old_log_probs).exp()

                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                pg_loss = torch.max(pg_loss1,pg_loss2).mean()

                v_loss = nn.MSELoss()(values_pred.squeeze(-1), b_returns)

                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_entropy += entropy.item()

        avg_pg_loss = total_pg_loss / 10
        avg_v_loss = total_v_loss / 10
        avg_entropy = total_entropy / 10

        return avg_pg_loss,avg_v_loss,avg_entropy,explained_var.item()

    def save_model(self,path,total_steps):
        torch.save({
            'policy_state_dict':self.policy.state_dict(),
            'optimizer_state_dict':self.optim.state_dict(),
            'total_steps':total_steps
        },path)
        print(f'模型已保存到{path},训练总步数为{total_steps}')

    def load_model(self,path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint.get('total_steps',0)
        print(f'模型文件{path}加载成功，总步数为{total_steps}')
        return total_steps
