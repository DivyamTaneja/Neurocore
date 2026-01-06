import torch
import numpy as np
import torch.nn.functional as F

class OneSpikeLIFLite:
    """Non-Pytorch version of your OneSpikeLIFNode, NO gradient."""
    def __init__(self, shape, threshold=0.5, reset=0.0, tau=2.0):
        self.v = np.zeros(shape, dtype=np.float32)
        self.fired = np.zeros(shape, dtype=np.bool_)
        self.threshold = threshold
        self.reset = reset
        self.tau = tau

    def step(self, x):
        """
        x: numpy array same shape as membrane
        returns spike np array {0,1}
        """

        # integrator: v = v + (x - v)/tau
        self.v = self.v + (x - self.v) / self.tau

        # check threshold
        spike = (self.v > self.threshold) & (~self.fired)

        # update fired mask
        self.fired |= spike

        # reset membrane where spiked
        self.v = np.where(spike, self.reset, self.v)

        return spike.astype(np.float32)


class SimplifiedSNN:
    """
    Fixed: compute conv+pool sizes correctly and use (H,W) ordering consistently.
    """

    def __init__(self, trained_model, H, W):
        self.H = H
        self.W = W

        # extract conv weights from real model
        self.W1 = trained_model.conv1.conv.weight.detach().cpu().numpy()  # shape [Cout1, Cin, kh, kw]
        self.b1 = trained_model.conv1.conv.bias.detach().cpu().numpy()

        self.W2 = trained_model.conv2.conv.weight.detach().cpu().numpy()  # shape [Cout2, Cin1, kh, kw]
        self.b2 = trained_model.conv2.conv.bias.detach().cpu().numpy()

        # conv output channels
        self.C1 = self.W1.shape[0]
        self.C2 = self.W2.shape[0]

        # kernel sizes (support non-square kernels too)
        kh1, kw1 = self.W1.shape[2], self.W1.shape[3]
        kh2, kw2 = self.W2.shape[2], self.W2.shape[3]

        # compute conv output sizes (valid conv), then pooling 2x2
        H_conv1 = self.H - kh1 + 1
        W_conv1 = self.W - kw1 + 1
        H1 = H_conv1 // 2
        W1 = W_conv1 // 2

        H_conv2 = H1 - kh2 + 1
        W_conv2 = W1 - kw2 + 1
        H2 = H_conv2 // 2
        W2 = W_conv2 // 2

        # LIF neurons for each layer: shapes based on conv->pool computed sizes
        self.neuron1 = OneSpikeLIFLite((self.C1, H1, W1))
        self.neuron2 = OneSpikeLIFLite((self.C2, H2, W2))

    def reset(self):
        # keep same shapes when resetting
        self.neuron1 = OneSpikeLIFLite(self.neuron1.v.shape)
        self.neuron2 = OneSpikeLIFLite(self.neuron2.v.shape)

    def conv2d(self, x, W, b, stride=1):
        # x: [Cin, H, W], W: [Cout, Cin, kh, kw]
        Cout, Cin, kh, kw = W.shape
        H, W0 = x.shape[1], x.shape[2]
        Ho = (H - kh) // stride + 1
        Wo = (W0 - kw) // stride + 1
        out = np.zeros((Cout, Ho, Wo), dtype=np.float32)

        for co in range(Cout):
            acc = np.zeros((Ho, Wo), dtype=np.float32)
            for ci in range(Cin):
                kern = W[co, ci]
                # slide kernel over input channel ci
                for i in range(Ho):
                    for j in range(Wo):
                        patch = x[ci, i:i+kh, j:j+kw]
                        acc[i, j] += np.sum(patch * kern)
            out[co] = acc + b[co]
        return out

    def pool2(self, x):
        C, H, W = x.shape
        Ho, Wo = H // 2, W // 2
        out = np.zeros((C, Ho, Wo), dtype=np.float32)
        for c in range(C):
            for h in range(Ho):
                for w in range(Wo):
                    blk = x[c, h*2:h*2+2, w*2:w*2+2]
                    out[c, h, w] = np.mean(blk)
        return out

    def process_spikes(self, layer, spike_list):
        if len(spike_list) == 0:
            return []

        # ---------- layer 0 ----------
        if layer == 0:
            C = 3
            # use (C, H, W) ordering
            dense = np.zeros((C, self.H, self.W), dtype=np.float32)
            for spike in spike_list:
                if len(spike) == 2:
                    x, y = spike
                    ch = 0
                else:
                    x, y, ch = spike
                # guard bounds
                if 0 <= y < self.H and 0 <= x < self.W and 0 <= ch < C:
                    dense[ch, y, x] = 1.0

            # conv1 -> pool -> LIF1
            x1 = self.conv2d(dense, self.W1, self.b1)
            x1 = self.pool2(x1)
            spikes1 = self.neuron1.step(x1)

            result = []
            C1, H1, W1 = spikes1.shape
            for c in range(C1):
                ys, xs = np.where(spikes1[c] > 0)
                for yy, xx in zip(ys, xs):
                    result.append((xx, yy, c))
            return result

        # ---------- layer 1 ----------
        elif layer == 1:
            # dense shape matches neuron1.v (C1,H1,W1)
            C1, H1, W1 = self.neuron1.v.shape
            dense = np.zeros((C1, H1, W1), dtype=np.float32)
            for spike in spike_list:
                x, y, ch = spike
                if 0 <= ch < C1 and 0 <= y < H1 and 0 <= x < W1:
                    dense[ch, y, x] = 1.0

            x2 = self.conv2d(dense, self.W2, self.b2)
            x2 = self.pool2(x2)
            spikes2 = self.neuron2.step(x2)

            result = []
            C2, H2, W2 = spikes2.shape
            for c in range(C2):
                ys, xs = np.where(spikes2[c] > 0)
                for yy, xx in zip(ys, xs):
                    result.append((xx, yy, c))
            return result

        else:
            return []
