# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.policy = nn.Linear(hidden, act_dim)
        self.value  = nn.Linear(hidden, 1)

    def forward(self, obs):
        z = self.backbone(obs)
        logits = self.policy(z)
        value = self.value(z).squeeze(-1)
        return logits, value

class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, clip=0.2, vf_coef=0.5, ent_coef=0.01):
        self.net = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.net(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item(), value.item()

    def evaluate_actions(self, obs_batch, act_batch):
        logits, values = self.net(obs_batch)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(act_batch)
        entropy = dist.entropy().mean()
        return logp, entropy, values

    def update(self, obs_buf, act_buf, adv_buf, ret_buf, old_logp_buf, epochs=4, minibatch=64):
        obs = torch.tensor(obs_buf, dtype=torch.float32)
        acts = torch.tensor(act_buf, dtype=torch.long)
        advs = torch.tensor(adv_buf, dtype=torch.float32)
        rets = torch.tensor(ret_buf, dtype=torch.float32)
        old_logp = torch.tensor(old_logp_buf, dtype=torch.float32)

        n = obs.shape[0]
        inds = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, n, minibatch):
                mb = inds[start:start+minibatch]
                mb_obs = obs[mb]
                mb_acts = acts[mb]
                mb_advs = advs[mb]
                mb_rets = rets[mb]
                mb_oldlog = old_logp[mb]

                logp, entropy, values = self.evaluate_actions(mb_obs, mb_acts)
                ratio = torch.exp(logp - mb_oldlog)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((mb_rets - values)**2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
