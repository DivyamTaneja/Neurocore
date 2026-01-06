# train.py
import os
import numpy as np
import torch
from torchvision import datasets, transforms

from env import SpikeControllerEnv
from ppo_agent import PPO

# Path to your trained PyTorch SNN weights (optional)
WEIGHT_PATH = "ttfs_based_scnn_model_weights.pth"

# ---------------------------
# Helper: build TTFS t-lists
# ---------------------------
def img_to_tlists(img_tensor, Tmax=8):
    """
    img_tensor: torch.Tensor shape [3,H,W], values in [0,1]
    Returns: t_lists_R, t_lists_G, t_lists_B where each is list length Tmax,
             each element is list of (x,y) coordinates whose timestep == t
    """
    C, H, W = img_tensor.shape
    assert C == 3
    R = img_tensor[0].numpy()
    G = img_tensor[1].numpy()
    B = img_tensor[2].numpy()

    t_lists_R = [[] for _ in range(Tmax)]
    t_lists_G = [[] for _ in range(Tmax)]
    t_lists_B = [[] for _ in range(Tmax)]

    for y in range(H):
        for x in range(W):
            # compute timesteps per channel
            tR = int(round((1.0 - float(R[y, x])) * (Tmax - 1)))
            tG = int(round((1.0 - float(G[y, x])) * (Tmax - 1)))
            tB = int(round((1.0 - float(B[y, x])) * (Tmax - 1)))

            # clamp safety
            tR = max(0, min(Tmax - 1, tR))
            tG = max(0, min(Tmax - 1, tG))
            tB = max(0, min(Tmax - 1, tB))

            t_lists_R[tR].append((x, y))
            t_lists_G[tG].append((x, y))
            t_lists_B[tB].append((x, y))

    return t_lists_R, t_lists_G, t_lists_B


# ---------------------------
# GAE helper (for PPO)
# ---------------------------
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    nextval = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        nextval = values[t]
    returns = adv + values
    return adv, returns


# ---------------------------
# Main training function
# ---------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # Hyperparams (small for quick testing)
    Tmax = 8
    H = 32
    W = 32
    epochs = 30
    steps_per_epoch = 512
    proc_per_step = 256

    # Load CIFAR sample (index 7 as before)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    img, label = train_dataset[7]
    img_tensor = img  # shape [3,32,32], values in [0,1]
    print("Loaded CIFAR sample index 7, label:", label)

    # Build t_lists for R,G,B channels
    tR, tG, tB = img_to_tlists(img_tensor, Tmax=Tmax)
    # quick counts
    total_spikes = sum(len(lst) for lst in tR) + sum(len(lst) for lst in tG) + sum(len(lst) for lst in tB)
    print(f"Total input spikes across channels: {total_spikes}")

    # Load trained PyTorch model (optional)
    trained_model = None
    if os.path.exists(WEIGHT_PATH):
        try:
            from snn_model import SCNN_CIFAR10_TTFS
            trained_model = SCNN_CIFAR10_TTFS()
            trained_model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'), strict=False)
            trained_model.eval()
            print("Loaded trained model weights from", WEIGHT_PATH)
        except Exception as e:
            print("Warning: couldn't load trained model weights:", e)
            trained_model = None
    else:
        print("No weight file found at", WEIGHT_PATH, "→ using random kernels in simulator")

    # Create environment (conv-only)
    env = SpikeControllerEnv(trained_model=trained_model, Tmax=Tmax, H=H, W=W)

    # Build action_map: flatten all valid (layer,t,row) combos to an integer index
    action_map = []
    for L in [0, 1]:
        rows = H if L == 0 else (H // 2)
        for t in range(Tmax):
            for r in range(rows):
                action_map.append((L, t, r))
    action_count = len(action_map)
    print("Action count (layer,t,row combos):", action_count)

    # Give env the initial spikes (it expects load_input_spikes)
    # Our SpikeControllerEnv expects three lists t_lists_R/G/B (per earlier design)
    env.reset()
    env.load_input_spikes(tR, tG, tB)  # push sample spikes into layer 0 memory

    # Observation dimension
    obs = env.get_obs()
    obs_dim = int(obs.shape[0])
    act_dim = action_count

    # Create PPO agent (imported from ppo_agent.py)
    agent = PPO(obs_dim, act_dim)

    # Training loop
    for epoch in range(epochs):
        # Storage buffers
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        logp_buf = []
        done_buf = []

        step = 0
        ep_rewards = []
        while step < steps_per_epoch:
            obs = env.get_obs()
            action_idx, logp, value = agent.select_action(obs)

            # map index -> (L,t,row)
            act_tuple = action_map[action_idx]

            # call env.step with tuple (it accepts tuple in conv-only design)
            # but our env implementation might accept integer or tuple; pass tuple
            next_obs, reward, done, info = env.step(act_tuple)

            # store transition
            obs_buf.append(obs.copy())
            act_buf.append(action_idx)
            rew_buf.append(reward)
            val_buf.append(value)
            logp_buf.append(logp)
            done_buf.append(done)

            step += 1

            if done:
                # episode finished (all spikes processed)
                ep_rewards.append(sum(rew_buf[-(step):]) if len(rew_buf) else 0.0)
                # reset environment and reload the same image spikes for next episode
                env.reset()
                env.load_input_spikes(tR, tG, tB)

        # compute advantages and returns
        advs, rets = compute_gae(rew_buf, val_buf, done_buf)
        # normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # update policy
        agent.update(np.array(obs_buf), np.array(act_buf), advs, rets, np.array(logp_buf), epochs=4, minibatch=128)

        # logging
        avg_reward = np.mean(rew_buf) if len(rew_buf) else 0.0
        pending_norm = env.get_obs()[0] if env.get_obs().size > 0 else 0.0
        print(f"[Epoch {epoch}] avg_reward={avg_reward:.4f} pending_norm={pending_norm:.4f}")

        # reload spikes so next epoch starts from same image
        env.reset()
        env.load_input_spikes(tR, tG, tB)

    # save policy network
    torch.save(agent.net.state_dict(), "ppo_policy_conv_only.pth")
    print("Saved policy to ppo_policy_conv_only.pth")


if __name__ == "__main__":
    main()
