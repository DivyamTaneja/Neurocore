# manual_tinker.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


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

img, label = train_dataset[14]
print("True label:", label)

img_tensor = img.unsqueeze(0)   # [1,C,H,W]
C, H, W = img.shape
print("Loaded image shape:", (C, H, W))

# -----------------------------
# 1) TTFS encoding
# -----------------------------
ttfs = TTFS_Encoder(T=Tmax)
spike_seq = ttfs(img_tensor)    # [1,T,C,H,W]
spike_seq = spike_seq.squeeze(0).cpu().numpy()  # [T,C,H,W]

print("Spike sequence shape:", spike_seq.shape)

for t in range(Tmax):
    spikes_t = spike_seq[t].sum()
    print(f"Spikes at t={t}: {int(spikes_t)}")


# -----------------------------
# 2) Plot original + spikes
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np

T, C, H, W = spike_seq.shape

# Function to produce a 2D map of spikes at timestep t
def spikes_2d_map(spike_seq_t):
    """
    spike_seq_t: [C, H, W] for one timestep
    Returns: [H, W] binary map showing if any channel spiked at each pixel
    """
    # Any channel spiking at a pixel => 1
    return (spike_seq_t.sum(axis=0) > 0).astype(float)

# Plot original image + spikes per timestep
plt.figure(figsize=(3 * (T + 1), 3))

# Original image
plt.subplot(1, T + 1, 1)
plt.imshow(np.transpose(img.numpy(), (1,2,0)))
plt.title("Original Image")
plt.axis("off")

# Spike maps for each timestep
for t in range(T):
    spike_map = spikes_2d_map(spike_seq[t])
    plt.subplot(1, T + 1, t + 2)
    plt.imshow(spike_map, cmap='gray')
    plt.title(f"t={t}")
    plt.axis("off")

plt.suptitle("Original Image vs Pixel-wise Spikes")
plt.tight_layout()
plt.show()
