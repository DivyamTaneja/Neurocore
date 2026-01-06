# manual_tinker.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from spike_memory import SpikeMemory
from neuron_memory import NeuronMemory
from spike_processor import SpikeProcessor
from snn_model import SCNN_CIFAR10_TTFS, TTFS_Encoder

# -----------------------------
# Config
# -----------------------------
Tmax = 2
NUM_LAYERS = 5   # 0=input,1=conv1,2=conv2,3=fc1,4=fc2

# -----------------------------
# 1) Load CIFAR image
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset  = datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

img, label = train_dataset[30]
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

# -----------------------------
# 6) Time-major simulation
# -----------------------------
print("\n=== Starting time-major simulation ===")

for t in range(Tmax):
    print(f"\n--- t = {t} ---")

    # reset fired flags each timestep (one-spike neurons)
    # for L in (1, 2, 3):
    #     neuron_mem.reset_fired(L)

    for layer in range(4):

        # -----------------------------
        # Conv layers (row-wise)
        # -----------------------------
        if layer == 0:
            _, H_in, _ = processor.shapes[layer]

            for r in range(H_in):
                row_spikes = spike_mem.get_spikes(layer, t, r)
                if not row_spikes:
                    continue

                processor.process_row_spikes(
                    layer=layer,
                    row_idx=r,
                    row_spikes=row_spikes
                )

            C_out, H_out, W_out = processor.shapes[layer + 1]

            for r in range(H_out):
                # processor.inject_bias(layer+1, r)
                # processor.inject_affine(layer+1, r)
            # Bias is ALREADY inside Conv2d → do NOT inject
                processor.commit_layer(layer + 1, r, t)

            # Collect spikes
            # C_out, H_out, W_out = processor.shapes[layer + 1]
            # # manual_spikes = np.zeros((C_out, H_out, W_out), dtype=np.float32)
            # for ch in range(C_out):
            #     for r in range(H_out):
            #         for c in range(W_out):
            #             if neuron_mem.get_fired(layer + 1, ch, r, c):
            #                 spike_mem.put_spike(
            #                     layer=layer + 1,
            #                     t=t,
            #                     ch=ch,
            #                     row=r,
            #                     col=c
            #                 )
            #                 # manual_spikes[ch,r,c] = 1
            # # print(manual_spikes)

        elif layer == 1:
            _, H_in, _ = processor.shapes[layer]

            for r in range(H_in):
                row_spikes = spike_mem.get_spikes(layer, t, r)
                if not row_spikes:
                    continue

                processor.process_row_spikes(
                    layer=layer,
                    row_idx=r,
                    row_spikes=row_spikes
                )

            C_out, H_out, W_out = processor.shapes[layer + 1]

            for r in range(H_out):
                # processor.inject_bias(layer+1, r)
                # processor.inject_affine(layer+1, r)
            # Bias is ALREADY inside Conv2d → do NOT inject
                processor.commit_layer(layer + 1, r, t)

            # Collect spikes
            # C_out, H_out, W_out = processor.shapes[layer + 1]
            # # manual_spikes = np.zeros((C_out, H_out, W_out), dtype=np.float32)
            # for ch in range(C_out):
            #     for r in range(H_out):
            #         for c in range(W_out):
            #             if neuron_mem.get_fired(layer + 1, ch, r, c):
            #                 spike_mem.put_spike(
            #                     layer=layer + 1,
            #                     t=t,
            #                     ch=0,
            #                     row=0,
            #                     col=ch * (H_out * W_out) + r * W_out + c
            #                 )
            #                 # manual_spikes[ch,r,c] = 1
            # # print(manual_spikes)

        # -----------------------------
        # FC layers
        # -----------------------------
        elif layer == 2:
            flat_spikes = spike_mem.get_spikes(layer, t, 0)
            # print("Layer - ", layer)
            # print(flat_spikes)
            # exit()
            if not flat_spikes:
                continue

            # print("Flat for layer = ", layer)
            processor.process_row_spikes(
                layer=layer,
                row_idx=0,
                row_spikes=flat_spikes
            )

            # processor.inject_bias(layer+1, 0)
            # processor.inject_affine(layer+1, 0)

            processor.commit_layer(layer + 1, 0, t)

            # for idx in range(processor.shapes[layer + 1][0]):
            #     if neuron_mem.get_fired(layer + 1, 0, 0, idx):
            #         spike_mem.put_spike(
            #             layer=layer + 1,
            #             t=t,
            #             ch=0,
            #             row=0,
            #             col=idx
            #         )

        elif layer == 3:
            flat_spikes = spike_mem.get_spikes(layer, t, 0)
            # print("Layer - ", layer)
            # print(flat_spikes)
            # exit()
            if not flat_spikes:
                continue

            # print("Flat for layer = ", layer)
            processor.process_row_spikes(
                layer=layer,
                row_idx=0,
                row_spikes=flat_spikes
            )

            # processor.inject_bias(layer + 1, 0)
            # processor.inject_affine(layer + 1, 0)

            processor.commit_layer(layer + 1, 0, t)

            # for idx in range(processor.shapes[layer + 1][0]):
            #     if neuron_mem.get_fired(layer + 1, 0, 0, idx):
            #         spike_mem.put_spike(
            #             layer=layer + 1,
            #             t=t,
            #             ch=0,
            #             row=0,
            #             col=idx
            #         )


print("\n=== Simulation finished ===")

# -----------------------------
# 7) Collect output spikes
# -----------------------------
outN = 10
out_matrix = np.zeros((Tmax, outN), dtype=float)

print("\nFinal layer-4 spikes:")
for t in range(Tmax):
    fs = spike_mem.get_spikes(4, t, 0)
    # print(f"t={t} :", fs)
    for ( _, idx) in fs:
        out_matrix[t, idx] = 1.0

print("\nFinal output spike matrix [T x 10]:")
print(out_matrix)

# -----------------------------
# 8) Decode prediction
# -----------------------------
spike_counts = out_matrix.sum(axis=0)
pred = int(np.argmax(spike_counts)) if spike_counts.sum() > 0 else -1

print("\nSpike counts per class:", spike_counts)
print("Predicted label:", pred, "| True label:", label)
