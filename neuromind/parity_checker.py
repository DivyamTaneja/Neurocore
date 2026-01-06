# parity_checker_ttfs.py
import torch
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from torchvision import datasets, transforms

from snn_model import SCNN_CIFAR10_TTFS, TTFS_Encoder
from spike_processor import SpikeProcessor
from neuron_memory import NeuronMemory

# -----------------------------
# 1) Load CIFAR sample
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
img, label = ds[7]
img = img.unsqueeze(0)  # [1,3,32,32]
print("Label:", label)

# -----------------------------
# 2) Load model
# -----------------------------
model = SCNN_CIFAR10_TTFS()
model.load_state_dict(
    torch.load("ttfs_based_scnn_model_weights.pth", map_location="cpu"),
    strict=False
)
model.eval()

# -----------------------------
# 3) Encode TTFS spikes
# -----------------------------
Tmax = 10
ttfs_encoder = TTFS_Encoder(T=Tmax)
spike_seq = ttfs_encoder(img)  # [B,T,C,H,W]

# t=0 spikes
spikes_t0 = spike_seq[:, 0]  # [B,C,H,W]


# -----------------------------
# 4) PyTorch reference (conv → conv → fc)
# -----------------------------
with torch.no_grad():
    # ---- Layer 1 ----
    x1 = model.pool1(model.conv1(spikes_t0))
    s1 = model.neuron1(x1)

    # ---- Layer 2 ----
    x2 = model.pool2(model.conv2(s1))
    s2 = model.neuron2(x2)

    # ---- FC ----
    x2_flat = s2.view(s2.size(0), -1)
    fc_out = model.fc1(x2_flat)
    s3 = model.neuron3(fc_out)   # <-- adjust name if neuron_fc1

    ref_flat = x2_flat.squeeze(0).cpu().numpy()
    ref_spikes_l2 = s2.squeeze(0).cpu().numpy()
    ref_spikes_fc = s3.squeeze(0).cpu().numpy()



# print("Reference membrane shape:", ref_mem.shape)
# print("Reference spike map shape:", ref_spikes.shape)

# -----------------------------
# 5) Manual spike-driven computation
# -----------------------------
neuron_mem = NeuronMemory()
processor = SpikeProcessor(model, neuron_mem)

# Reset neuron memory
neuron_mem.reset_all()

# Convert t=0 spikes to numpy
x_np = spikes_t0.squeeze(0).cpu().numpy()  # [C,H,W]
C_in, H, W = x_np.shape

# Row-wise spike processing (t=0)
for ch in range(C_in):
    for r in range(H):
        row_spikes = [(ch, c) for c in range(W) if x_np[ch, r, c] != 0.0]
        if row_spikes:
            processor.process_row_spikes(layer=0, row_idx=r, row_spikes=row_spikes)

# # Inject bias once after conv+pool
print("injecting bias")
processor.inject_bias(layer=1)
processor.inject_affine(layer=1)
# # Commit LIF update
processor.commit_layer(layer=1)

C1, H1, W1 = processor.shapes[1]
layer1_spikes = []

for ch in range(C1):
    for r in range(H1):
        for c in range(W1):
            if neuron_mem.get_fired(1, ch, r, c):
                layer1_spikes.append((ch, r, c))

for (ch, r, c) in layer1_spikes:
    processor.process_row_spikes(
        layer=1,
        row_idx=r,
        row_spikes=[(ch, c)]
    )

# Bias + affine
processor.inject_bias(layer=2)
processor.inject_affine(layer=2)

# Commit neuron2
processor.commit_layer(layer=2)


C2, H2, W2 = processor.shapes[2]
layer2_spikes = []

for ch in range(C2):
    for r in range(H2):
        for c in range(W2):
            if neuron_mem.get_fired(2, ch, r, c):
                layer2_spikes.append((ch, r, c))

# Reset FC neuron memory
neuron_mem.reset_fired(3)

# Convert to flat spikes (same as SpikeMemory does)

# print(layer2_spikes)

flat_spikes = []
for (ch, r, c) in layer2_spikes:
    idx = ch * (H2 * W2) + r * W2 + c
    flat_spikes.append(idx)

# print(flat_spikes)
# Accumulate via SpikeProcessor (THIS is the parity path)
processor.process_row_spikes(
    layer=2,
    row_idx=0,          # unused for FC
    row_spikes=flat_spikes
)

# Bias + affine + LIF
processor.inject_bias(3)
processor.inject_affine(3)
processor.commit_layer(3)


# -----------------------------
# 6) Extract manual membrane & spike map
# -----------------------------
# C, H, W = processor.shapes[1]
# manual_mem = np.zeros((C,H,W), dtype=np.float32)
# manual_spikes = np.zeros((C,H,W), dtype=np.float32)

# for ch in range(C):
#     for r in range(H):
#         for c in range(W):
#             key = neuron_mem._key(1, ch, r, c)
#             manual_mem[ch,r,c] = neuron_mem.input_acc.get(key, 0.0)
#             manual_spikes[ch,r,c] = neuron_mem.get_fired(1,ch,r,c)

C2, H2, W2 = processor.shapes[2]
manual_spikes_l2 = np.zeros((C2, H2, W2), dtype=np.float32)

for ch in range(C2):
    for r in range(H2):
        for c in range(W2):
            manual_spikes_l2[ch, r, c] = neuron_mem.get_fired(2, ch, r, c)

diff = np.abs(ref_spikes_l2 - manual_spikes_l2)
print("\n--- NEURON2 SPIKE PARITY (t=0) ---")
print("Total mismatches:", diff.sum())

# -----------------------------
# 7) Compare membrane
# -----------------------------
# diff_mem = np.abs(ref_mem - manual_mem)
# print("\n--- NEURON1 MEMBRANE PARITY (t=0) ---")
# print("Max diff :", diff_mem.max())
# print("Mean diff:", diff_mem.mean())
# fm = 0
# print(f"Feature map {fm} max diff:", diff_mem[fm].max())

# # -----------------------------
# # 8) Compare spikes
# # -----------------------------
# diff_spikes = np.abs(ref_spikes - manual_spikes)
# print("\n--- NEURON1 SPIKE PARITY (t=0) ---")
# print("Total spike mismatches:", diff_spikes.sum())


# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# pool_out = pool_out.squeeze(0).cpu().numpy()

# # print("Ref pool out - ")
# # print(pool_out[0])

# # print("Ref manual - ")
# # print(manual_mem[0])

# diff_spikes = np.abs(ref_spikes - manual_spikes)
# print(diff_spikes.sum())

# print("Reference_spikes - ")
# print(ref_spikes[0])

# print("Manual_spikes - ")
# print(manual_spikes[0])

N3 = processor.shapes[3][0]
manual_spikes_l3 = np.zeros((N3,), dtype=np.float32)

for i in range(N3):
    manual_spikes_l3[i] = neuron_mem.get_fired(3, 0, 0, i)

# -----------------------------
# Parity check
# -----------------------------
diff = np.abs(ref_spikes_fc - manual_spikes_l3)

print("\n--- FC1 SPIKE PARITY (t=0) ---")
print("Total mismatches:", diff.sum())

if diff.sum() > 0:
    bad = np.where(diff != 0)[0]
    print("Mismatch indices:", bad[:10])