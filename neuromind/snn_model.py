import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

# class TTFS_Encoder:
#     def __init__(self, T=8):
#         self.T = T

#     def __call__(self, x):
#         spike_times = (1.0 - x) * (self.T - 1)
#         spike_tensor = torch.zeros(x.size(0), self.T, *x.shape[1:], device=x.device)
#         for t in range(self.T):
#             spike_tensor[:, t][(spike_times.round() == t)] = 1.0
#         return spike_tensor

import torch
import torch.nn.functional as F

class TTFS_Encoder:
    def __init__(self, T=2, patch=4):
        """
        T     : number of timesteps (2 or 3)
        patch : patch size (4x4 recommended for CIFAR)
        """
        self.T = T
        self.patch = patch

        # CIFAR-10 stats (constants)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
        self.std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

        # Thresholds (fixed)
        self.body_th   = 0.25    # body detection
        self.edge_th   = 0.40    # edge detection
        self.early_th  = 0.90    # very strong signal

    def __call__(self, x):
        """
        x: [B, C, H, W] in [0,1]
        returns: [B, T, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        mean = self.mean.to(device)
        std  = self.std.to(device)

        # Normalize
        x_norm = (x - mean) / std

        # Local contrast (edges)
        local_mean = F.avg_pool2d(
            x_norm, kernel_size=3, stride=1, padding=1
        )
        delta = x_norm - local_mean

        spikes = torch.zeros(B, self.T, C, H, W, device=device)

        # --------------------------------------------------
        # PATCH-BASED DECISION (core logic)
        # --------------------------------------------------
        for b in range(B):
            for i in range(0, H, self.patch):
                for j in range(0, W, self.patch):
                    patch_x = x_norm[b, :, i:i+self.patch, j:j+self.patch]
                    patch_d = delta[b, :, i:i+self.patch, j:j+self.patch]

                    # Representative signals
                    body_strength = patch_x.abs().mean()
                    edge_strength = patch_d.abs().max()

                    # Decide spike time
                    if body_strength > self.early_th or edge_strength > self.early_th:
                        t = 0
                    elif body_strength > self.body_th or edge_strength > self.edge_th:
                        t = 1
                    else:
                        continue  # no spike for this patch

                    # Emit ONE spike per patch (strongest pixel)
                    if edge_strength > body_strength:
                        # choose strongest edge pixel
                        idx = patch_d.abs().view(-1).argmax()
                    else:
                        # choose strongest body pixel
                        idx = patch_x.abs().view(-1).argmax()

                    c = idx // (self.patch * self.patch)
                    rem = idx % (self.patch * self.patch)
                    r = rem // self.patch
                    c2 = rem % self.patch

                    spikes[b, t, c, i+r, j+c2] = 1.0

        return spikes


    
class TemporalSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0, beta=5.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.beta = beta
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        threshold = ctx.threshold
        beta = ctx.beta
        grad_input = grad_output * beta * torch.exp(-beta * torch.abs(input - threshold))
        return grad_input, None, None


class OneSpikeLIFNode(nn.Module):
    """
    AMOS-style one-spike LIF neuron for **per-timestep** input:
        Input:  [B, C, H, W]  or [B, N]
        Output: same shape, spike tensor
    The neuron:
      • Integrates over time internally
      • Fires at most one spike
      • Uses surrogate gradient for differentiability
    """

    def __init__(self, v_threshold=0.5, v_reset=0.0, tau=2.0, beta=2.0):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau = tau
        self.beta = beta
        self.register_buffer("v", None)
        self.register_buffer("fired_mask", None)

    def reset(self):
        self.v = None
        self.fired_mask = None

    def forward(self, x):
        """
        x: [B, C, H, W] or [B, N] — single timestep input
        returns: same shape (spike output)
        """
        device = x.device

        # Initialize membrane and firing mask
        if self.v is None:
            self.v = torch.zeros_like(x, device=device)
            self.fired_mask = torch.zeros_like(x, device=device)

        # integrate membrane potential
        self.v = self.v + (x - self.v) / self.tau

        # surrogate spike generation
        spike = TemporalSurrogate.apply(self.v, self.v_threshold, self.beta)

        # only allow first spike
        new_spike = (1 - self.fired_mask) * spike

        # update firing state
        self.fired_mask = torch.clamp(self.fired_mask + new_spike, 0, 1)

        # reset potential where spiked
        self.v = torch.where(new_spike.bool(), torch.full_like(self.v, self.v_reset), self.v)

        return new_spike


# ------------------------------------------------------------
# ---- Helper functions implementing E-TTFS equations ----
# ------------------------------------------------------------
def ettfs_init_weights(layer, T):
    """Eq.(14): E-TTFS initialization."""
    if isinstance(layer, nn.Conv2d):
        N_in = layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
    elif isinstance(layer, nn.Linear):
        N_in = layer.weight.shape[1]
    else:
        return

    bound = (3 * T / N_in) ** 0.5
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


def ettfs_weight_norm(layer, T, eps=1e-8):
    """Eq.(16): layer-wise mean–variance normalization (E-TTFS)."""
    with torch.no_grad():
        W = layer.weight
        mean = W.mean()
        std = W.std()

        # Compute N_in (input features per neuron)
        if isinstance(layer, nn.Conv2d):
            N_in = W.shape[1] * W.shape[2] * W.shape[3]
        else:
            N_in = W.shape[1]

        # σ_Wl from Eq. (14)
        sigma_Wl = torch.sqrt(torch.tensor(3.0 * T / N_in, device=W.device, dtype=W.dtype))

        # Apply normalization
        W.copy_((W - mean) / (std + eps) * sigma_Wl)


# ------------------------------------------------------------
# ---- Convolution + affine transform block ----
# ------------------------------------------------------------
class ETTFSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, T, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=False)
        # self.gamma = nn.Parameter(torch.ones(1))
        # self.beta = nn.Parameter(torch.zeros(1))
        ettfs_init_weights(self.conv, T)
        ettfs_weight_norm(self.conv, T)

    def forward(self, x):
        x = self.conv(x)
        # Eq.(17): affine transform
        # return self.gamma * x + self.beta
        return x


# ------------------------------------------------------------
# ---- Linear + affine transform block ----
# ------------------------------------------------------------
class ETTFSLinearBlock(nn.Module):
    def __init__(self, in_f, out_f, T):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        # self.gamma = nn.Parameter(torch.ones(1))
        # self.beta = nn.Parameter(torch.zeros(1))
        ettfs_init_weights(self.fc, T)
        ettfs_weight_norm(self.fc, T)

    def forward(self, x):
        x = self.fc(x)
        # return self.gamma * x + self.beta
        return x
    
class SCNN_CIFAR10_TTFS(nn.Module):
    def __init__(self, T=2):
        super().__init__()
        self.T = T

        # CIFAR10 input = [B, 3, 32, 32]

        # ----- Layer definitions -----
        self.conv1 = ETTFSConvBlock(3, 32, T)    # output: [32, 32, 32]
        self.pool1 = nn.AvgPool2d(2, 2)          # -> [32, 16, 16]
        self.neuron1 = OneSpikeLIFNode()

        self.conv2 = ETTFSConvBlock(32, 64, T)   # -> [64, 16, 16]
        self.pool2 = nn.AvgPool2d(2, 2)          # -> [64, 8, 8]
        self.neuron2 = OneSpikeLIFNode()

        # Fully connected
        self.fc1 = ETTFSLinearBlock(64 * 8 * 8, 256, T)
        self.neuron3 = OneSpikeLIFNode()

        self.fc2 = ETTFSLinearBlock(256, 10, T)
        self.neuron4 = OneSpikeLIFNode()

    # ----------------------------------------------------
    def forward(self, x):
        # Encode static CIFAR-10 image to TTFS spike train
        spike_seq = TTFS_Encoder(T=self.T)(x)  # [B, T, 3, 32, 32]
        B = x.size(0)

        # Output spike recording
        out_spikes = torch.zeros(B, self.T, 10, device=x.device)

        # Reset neurons for temporal rollout
        for m in self.modules():
            if isinstance(m, OneSpikeLIFNode):
                m.reset()

        # Step-by-step temporal propagation
        for t in range(self.T):
            cur = spike_seq[:, t]      # [B, 3, 32, 32]

            x = self.conv1(cur)        # [B, 32, 32, 32]
            x = self.pool1(x)          # [B, 32, 16, 16]
            x = self.neuron1(x)

            x = self.conv2(x)          # [B, 64, 16, 16]
            x = self.pool2(x)          # [B, 64, 8, 8]
            x = self.neuron2(x)

            # Flatten for FC
            x = x.flatten(1)           # [B, 64*8*8]

            x = self.fc1(x)            # [B, 256]
            x = self.neuron3(x)

            x = self.fc2(x)            # [B, 10]
            x = self.neuron4(x)

            # Collect spikes at time t
            out_spikes[:, t] = x

        return out_spikes
    
class TemporalWeightingDecoder(nn.Module):
    """
    Implements Eqs. (20–24) from Efficient-TTFS.
    Decodes time-series spikes [B, T, C] into [B, C] logits.
    """

    def __init__(self, T, gamma=0.1, mode='exp'):
        """
        Args:
            T: number of timesteps
            gamma: scaling factor (>1)
            mode: 'exp' for exponential decay, 'linear' for linear decay
        """
        super().__init__()
        self.T = T
        self.gamma = gamma
        self.mode = mode
        self.register_buffer("weights", self._make_weights())

    def _make_weights(self):
        t = torch.arange(self.T, dtype=torch.float32)
        if self.mode == 'exp':
            # Eq. (23): exponential decay
            w = torch.pow(self.gamma, -t)
        elif self.mode == 'linear':
            # Eq. (24): linear decay
            w = self.gamma * (self.T - t) / self.T
        else:
            raise ValueError("mode must be 'exp' or 'linear'")
        # Normalize so that w[0] = 1 (for consistency)
        w = w / w[0]
        return w

    def forward(self, out_spikes):
        """
        Args:
            out_spikes: Tensor [B, T, C] of binary or analog spike outputs
        Returns:
            decoded_logits: Tensor [B, C]
        """
        # Eq. (20): Y = sum_t w[t] * O[t]
        decoded = torch.einsum('btc,t->bc', out_spikes, self.weights)
        return decoded