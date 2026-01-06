from controller import (
    ControllerState,
    get_action_space,
    execute_action
)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from spike_memory import SpikeMemory
from neuron_memory import NeuronMemory
from spike_processor import SpikeProcessor
from snn_model import SCNN_CIFAR10_TTFS, TTFS_Encoder


Tmax = 2
NUM_LAYERS = 5   # 0=input,1=conv1,2=conv2,3=fc1,4=fc2

# -----------------------------
# 1) Load CIFAR image
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

img, label = train_dataset[13]
print("True label:", label)

img_tensor = img.unsqueeze(0)   # [1,C,H,W]
C, H, W = img.shape
print("Loaded image shape:", (C, H, W))

# -----------------------------
# 2) TTFS encoding
# -----------------------------
ttfs = TTFS_Encoder(T=Tmax)
spike_seq = ttfs(img_tensor)    # [1,T,C,H,W]
spike_seq = spike_seq.squeeze(0).cpu().numpy()  # [T,C,H,W]

# -----------------------------
# 3) SpikeMemory
# -----------------------------
spike_mem = SpikeMemory(num_layers=NUM_LAYERS, Tmax=Tmax)
for L in range(NUM_LAYERS):
    spike_mem.reset_layer(L)

# Store input spikes (layer 0)
for t in range(Tmax):
    cur = spike_seq[t]
    for ch in range(C):
        for r in range(H):
            for c in range(W):
                if cur[ch, r, c] != 0.0:
                    spike_mem.put_spike(
                        layer=0, t=t, ch=ch, row=r, col=c
                    )

print("Input TTFS spikes loaded.")

# -----------------------------
# 4) Load trained model
# -----------------------------
model = SCNN_CIFAR10_TTFS()
model.load_state_dict(
    # torch.load("ttfs_based_scnn_model_weights.pth", map_location="cpu"),
    torch.load("ttfs_based_scnn_model_weights_new_encoder.pth", map_location="cpu"),
    strict=False
)
model.eval()

# -----------------------------
# 5) Processor + NeuronMemory
# -----------------------------
neuron_mem = NeuronMemory()

processor = SpikeProcessor(
    model,
    neuron_mem,
    spike_mem
)

print("SpikeProcessor initialized.")

state = ControllerState(processor, Tmax)

output_layer = max(processor.shapes.keys())
step = 0

while True:
    print(f"\n===== STEP {step} =====")

    # 1. Termination check
    if spike_mem.has_output_spike(output_layer):
        print("Output spike detected → STOP")
        row_spikes = spike_mem.get_spikes(output_layer, 0, 0)
        print(row_spikes)
        break

    print("Active spikes in SpikeMemory:", spike_mem.count_total_spikes())
    print("Active neuron memory:", neuron_mem.total_active())

    # 2. Query legal actions
    actions = get_action_space(state, processor)

    if not actions:
        print("Deadlock: no legal actions")
        break

    print("Legal actions:")
    zero_spike_actions = []

    for i, (L, R) in enumerate(actions):
        T = state.exec_t[(L, R)]
        n_spikes = spike_mem.count_row_spikes_at_t(L, T, R)

        print(
            f"[{i}] Layer={L}, Row={R}, "
            f"t={T}, row_spikes={n_spikes}"
        )

        if n_spikes == 0:
            zero_spike_actions.append((i, (L, R)))

    # --------------------------------------------------
    # 3. Auto-execute zero-spike action if available
    # --------------------------------------------------
    # if zero_spike_actions:
    #     i, action = zero_spike_actions[0]   # pick first (deterministic)
    #     print(
    #         f"Auto-executing zero-spike action "
    #         f"[{i}] Layer={action[0]}, Row={action[1]}"
    #     )
    # else:
    #     idx = int(input("Choose action index: "))
    #     action = actions[idx]

    idx = int(input("Choose action index: "))
    action = actions[idx]

    # 4. Execute exactly ONE step
    execute_action(action, state, processor, spike_mem, neuron_mem)

    step += 1
