# spike_memory.py
from typing import List, Tuple

class SpikeMemory:
    """
    Sparse spike memory for TTFS simulation.
    Structure:
        mem[layer][t][row] = list of (ch, col)
    Conv layers: store per-row for streaming.
    FC layers: use row==0 with (0, index).
    """

    def __init__(self, num_layers: int, Tmax: int):
        self.num_layers = int(num_layers)
        self.Tmax = int(Tmax)
        self.mem = {layer: {} for layer in range(self.num_layers)}

    def reset_layer(self, layer: int):
        self.mem[int(layer)] = {}

    def reset_all(self):
        self.mem = {layer: {} for layer in range(self.num_layers)}

    def put_spike(self, layer: int, t: int, ch: int, row: int, col: int):
        layer, t, ch, row, col = int(layer), int(t), int(ch), int(row), int(col)
        if layer not in self.mem:
            self.mem[layer] = {}
        if t not in self.mem[layer]:
            self.mem[layer][t] = {}
        if row not in self.mem[layer][t]:
            self.mem[layer][t][row] = []
        self.mem[layer][t][row].append((ch, col))

    def get_spikes(self, layer: int, t: int, row: int) -> List[Tuple[int,int]]:
        """Return list[(ch,col)] for (layer,t,row) and consume them."""
        layer, t, row = int(layer), int(t), int(row)
        if layer not in self.mem:
            return []
        if t not in self.mem[layer]:
            return []
        if row not in self.mem[layer][t]:
            return []
        spikes = self.mem[layer][t][row]
        # consume row
        del self.mem[layer][t][row]
        if not self.mem[layer][t]:
            del self.mem[layer][t]
        return [(int(ch), int(col)) for (ch, col) in spikes]

    def count_total_spikes(self) -> int:
        total = 0
        for layer in self.mem:
            for t in self.mem[layer]:
                for row in self.mem[layer][t]:
                    total += len(self.mem[layer][t][row])
        return total

    def load_from_ttfs_tensor(self, spike_tensor, layer=0):
        """
        Helper: fill a layer from TTFS output tensor (T, C, H, W).
        Stores spikes into mem[layer] as (ch,row,col) for each t.
        """
        import numpy as _np
        import torch as _torch
        arr = spike_tensor
        if _torch.is_tensor(arr):
            arr = arr.cpu().numpy()
        T, C, H, W = arr.shape
        self.mem[layer] = {}
        for t in range(T):
            self.mem[layer][t] = {}
            frame = arr[t]
            nz = _np.argwhere(frame != 0)
            for (ch, r, c) in nz:
                if r not in self.mem[layer][t]:
                    self.mem[layer][t][r] = []
                self.mem[layer][t][r].append((int(ch), int(c)))

    def has_output_spike(self, layer: int) -> bool:
        """
        Returns True if any spike exists in the given output layer
        at any timestep and any row.
        """
        layer = int(layer)
        if layer not in self.mem:
            return False

        for t in self.mem[layer]:
            if self.mem[layer][t]:   # any row has spikes
                return True
        return False
    
    def count_row_spikes_at_t(self, layer: int, t: int, row: int) -> int:
        layer, t, row = int(layer), int(t), int(row)

        if layer not in self.mem:
            return 0
        if t not in self.mem[layer]:
            return 0
        if row not in self.mem[layer][t]:
            return 0

        return len(self.mem[layer][t][row])
