# spike_processor.py
import math
import numpy as np


class SpikeProcessor:
    def __init__(self, model, neuron_mem, spike_mem):
        self.model = model
        self.neuron_mem = neuron_mem
        self.spike_mem = spike_mem

        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.fc1 = model.fc1
        self.fc2 = model.fc2

        # Output neuron shapes (after pool)
        self.shapes = {
            0: (3, 32, 32),      # input
            1: (32, 16, 16),     # conv1 + pool
            2: (64, 8, 8),       # conv2 + pool
            3: (256,),           # fc1
            4: (10,)             # fc2
        }

        # Explicit conv metadata (ONLY where needed)
        self.conv_params = {
            1: (3, 1, 1),  # conv1: K=3, S=1, P=1
            2: (3, 1, 1),  # conv2: K=3, S=1, P=1
        }

        self.pool_params = {
            1: (2,2), 
            2: (2,2)
        }
    
    # ------------------------------------------------------------
    # Neuron parameters
    # ------------------------------------------------------------
    def get_neuron_params(self, layer):
        if layer == 1:
            n = self.model.neuron1
        elif layer == 2:
            n = self.model.neuron2
        elif layer == 3:
            n = self.model.neuron3
        elif layer == 4:
            n = self.model.neuron4
        else:
            raise ValueError(f"Layer {layer} has no neuron params")

        return float(n.tau), float(n.v_threshold), float(n.v_reset)

    # ------------------------------------------------------------
    # Phase 1 — SPIKE-DRIVEN INPUT ACCUMULATION
    # ------------------------------------------------------------
    def process_row_spikes(self, layer, row_idx, row_spikes):
        """
        row_spikes: list of (channel, column)
        """
        if not row_spikes:
            return

        if layer == 0:
            # Input → Conv1 + Pool → Neuron1
            self._accumulate_conv_pool(
                row_idx, row_spikes,
                self.conv1.conv,
                next_layer=1
            )

        elif layer == 1:
            # Neuron1 → Conv2 + Pool → Neuron2
            self._accumulate_conv_pool(
                row_idx, row_spikes,
                self.conv2.conv,
                next_layer=2
            )

        elif layer == 2:
            self._accumulate_fc(row_spikes, self.fc1, next_layer=3)

        elif layer == 3:
            self._accumulate_fc(row_spikes, self.fc2, next_layer=4)

    # ------------------------------------------------------------
    # Conv + AvgPool (FUSED, NO BIAS)
    # ------------------------------------------------------------
    def _accumulate_conv_pool(self, row_idx, row_spikes, conv, next_layer):
        """
        Fully spike-driven Conv + AvgPool accumulation.
        Each input spike directly updates pooled neuron memory.
        Bias is NOT added here (inject separately).
        """

        # --------------------------------------------------
        # Load conv params
        # --------------------------------------------------
        W = conv.weight.detach().cpu().numpy()   # [C_out, C_in, K, K]
        C_out, C_in, K, _ = W.shape
        stride = conv.stride[0]
        pad = conv.padding[0]

        # --------------------------------------------------
        # Geometry
        # --------------------------------------------------
        _, H_in, W_in = self.shapes[next_layer - 1]

        H_conv = (H_in + 2 * pad - K) // stride + 1
        W_conv = (W_in + 2 * pad - K) // stride + 1

        pool_k = 2
        pool_s = 2
        pool_weight = 1.0 / (pool_k * pool_k)

        H_pool = (H_conv - pool_k) // pool_s + 1
        W_pool = (W_conv - pool_k) // pool_s + 1

        # --------------------------------------------------
        # Spike-driven accumulation
        # --------------------------------------------------
        for ch_in, c in row_spikes:
            spike_val = 1.0   # TTFS spike at t=0 (binary)

            # Find all conv outputs affected by (row_idx, c)
            i_min = int(np.ceil((row_idx + pad - (K - 1)) / stride))
            i_max = int((row_idx + pad) // stride)
            j_min = int(np.ceil((c + pad - (K - 1)) / stride))
            j_max = int((c + pad) // stride)

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    if not (0 <= i < H_conv and 0 <= j < W_conv):
                        continue

                    u = row_idx - (i * stride - pad)
                    v = c - (j * stride - pad)

                    if not (0 <= u < K and 0 <= v < K):
                        continue

                    # ------------------------------
                    # Pool window boundaries
                    # ------------------------------
                    pr = i // pool_s
                    pc = j // pool_s

                    # Ensure this (i,j) truly lies inside the pool window
                    if not (
                        pr * pool_s <= i < pr * pool_s + pool_k and
                        pc * pool_s <= j < pc * pool_s + pool_k
                    ):
                        continue

                    if not (0 <= pr < H_pool and 0 <= pc < W_pool):
                        continue

                    # ------------------------------
                    # Accumulate into pooled neuron
                    # ------------------------------
                    for co in range(C_out):
                        self.neuron_mem.add_input(
                            next_layer,
                            co, pr, pc,
                            spike_val * W[co, ch_in, u, v] * pool_weight
                        )


    # ------------------------------------------------------------
    # Fully-connected accumulation (bias injected separately)
    # ------------------------------------------------------------
    def _accumulate_fc(self, row_spikes, fc, next_layer):
        W = fc.fc.weight.detach().cpu().numpy()
    
        for s in row_spikes:
            idx = s if isinstance(s, int) else s[-1]

            for o in range(W.shape[0]):
                self.neuron_mem.add_input(
                    next_layer, 0, 0, o,
                    float(W[o, idx])
                )

    # ------------------------------------------------------------
    # Bias injection — ONCE PER TIMESTEP
    # ------------------------------------------------------------
    def inject_bias(self, layer: int, row: int):
        if layer == 1:
            bias = self.conv1.conv.bias.detach().cpu().numpy()
        elif layer == 2:
            bias = self.conv2.conv.bias.detach().cpu().numpy()
        elif layer == 3:
            bias = self.fc1.fc.bias.detach().cpu().numpy()
        elif layer == 4:
            bias = self.fc2.fc.bias.detach().cpu().numpy()
        else:
            return

        shape = self.shapes[layer]

        if len(shape) == 3:
            C, H, W = shape
            if not (0 <= row < H):
                print("lafda")
                return

            for ch in range(C):
                for c in range(W):
                    self.neuron_mem.add_input(
                        layer, ch, row, c, float(bias[ch])
                    )
        else:
            for i in range(shape[0]):
                # print("Before bias - ", self.neuron_mem.input_acc.get(key, 0.0))
                self.neuron_mem.add_input(
                    layer, 0, 0, i, float(bias[i])
                )
                # print("After bias - ", self.neuron_mem.input_acc.get(key, 0.0))
                
    # ------------------------------------------------------------
    # AFFINE TRANSFORM (gamma * x + beta)
    # ------------------------------------------------------------
    def inject_affine(self, layer: int, row: int):
        if layer == 1:
            block = self.conv1
        elif layer == 2:
            block = self.conv2
        elif layer == 3:
            block = self.fc1
        elif layer == 4:
            block = self.fc2
        else:
            return

        gamma = float(block.gamma.detach().cpu().item())
        beta = float(block.beta.detach().cpu().item())

        shape = self.shapes[layer]

        if len(shape) == 3:
            C, H, W = shape
            if not (0 <= row < H):
                print("lafda")
                return

            for ch in range(C):
                for c in range(W):
                    key = self.neuron_mem._key(layer, ch, row, c)
                    if key not in self.neuron_mem.input_acc:
                        continue
                    x = self.neuron_mem.input_acc[key]
                    self.neuron_mem.input_acc[key] = gamma * x + beta
        else:
            for i in range(shape[0]):
                key = self.neuron_mem._key(layer, 0, 0, i)
                x = self.neuron_mem.input_acc.get(key, 0.0)
                self.neuron_mem.input_acc[key] = gamma * x + beta
                x = self.neuron_mem.input_acc.get(key, 0.0)

    # ------------------------------------------------------------
    # Phase 2 — LIF COMMIT
    # ------------------------------------------------------------
    def commit_layer(self, layer: int, row: int, t: int):

        tau, v_thresh, v_reset = self.get_neuron_params(layer)
        # print("yes")
        self.neuron_mem.commit_layer(
            layer, row, tau, v_thresh, v_reset
        )

        if layer == 1:
            C_out, H_out, W_out = self.shapes[layer]
            for ch in range(C_out):
                for col in range(W_out):
                    if self.neuron_mem.get_fired(layer, ch, row, col):
                        self.spike_mem.put_spike(
                            layer=layer,
                            t=t,
                            ch=ch,
                            row=row,
                            col=col
                        )

        elif layer == 2:
            C_out, H_out, W_out = self.shapes[layer]
            for ch in range(C_out):
                for col in range(W_out):
                    if self.neuron_mem.get_fired(layer, ch, row, col):
                        self.spike_mem.put_spike(
                            layer=layer,
                            t=t,
                            ch=0,
                            row=0,
                            col=ch * (H_out * W_out) + row * W_out + col
                        )

        elif layer == 3:
            W_out = self.shapes[layer][0]
            for col in range(W_out):
                if self.neuron_mem.get_fired(layer, 0, 0, col):
                    self.spike_mem.put_spike(
                        layer=layer,
                        t=t,
                        ch=0,
                        row=0,
                        col=col
                    )

        elif layer == 4:
            W_out = self.shapes[layer][0]
            for col in range(W_out):
                if self.neuron_mem.get_fired(layer, 0, 0, col):
                    self.spike_mem.put_spike(
                        layer=layer,
                        t=t,
                        ch=0,
                        row=0,
                        col=col
                    )