# neuron_memory.py
from typing import Iterator, Tuple


class NeuronMemory:
    """
    Sparse neuron memory with:
      - input accumulator (pre-LIF)
      - membrane potential
      - fired flag
    """

    def __init__(self):
        self.vmem = {}        # (layer,ch,row,col) -> float
        self.fired = {}       # (layer,ch,row,col) -> 1
        self.masked = {}      # permanent "already fired"
        self.input_acc = {}   # (layer,ch,row,col) -> float

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _key(self, layer: int, ch: int, row: int, col: int) -> Tuple[int, int, int, int]:
        return (int(layer), int(ch), int(row), int(col))

    # --------------------------------------------------
    # Phase 1 — Input accumulation
    # --------------------------------------------------
    def add_input(self, layer: int, ch: int, row: int, col: int, val: float):
        key = self._key(layer, ch, row, col)
        # print("Previous", self.input_acc.get(key, 0.0))
        if self.masked.get(key, 0):
            return

        self.input_acc[key] = self.input_acc.get(key, 0.0) + float(val)
        # print("Current", self.input_acc.get(key, 0.0))

    # --------------------------------------------------
    # Phase 2 — LIF commit (ONCE per timestep)
    # --------------------------------------------------
    def commit_layer(self, layer: int, row: int, tau: float, v_thresh: float, v_reset: float):
        """
        Apply LIF update ONCE using accumulated inputs.
        """
        keys = [k for k in self.input_acc.keys() if k[0] == layer and (row is None or k[2] == row)]

        for (L, ch, r, c) in keys:
            inp = self.input_acc[(L, ch, r, c)]
            v_prev = self.vmem.get((L, ch, r, c), 0.0)
            masked = self.masked.get((L, ch, r, c), 0)

            # Standard discrete LIF update
            v_new = v_prev + (inp - v_prev) / tau

            if v_new > v_thresh and not masked:
                self.fired[(L, ch, r, c)] = 1
                self.masked[(L, ch, r, c)] = 1
                self.vmem[(L, ch, r, c)] = v_reset
            else:
                if abs(v_new) < 1e-8:
                    self.vmem.pop((L, ch, r, c), None)
                else:
                    self.vmem[(L, ch, r, c)] = v_new

        # Clear accumulated inputs for this layer
        self.clear_input_acc(layer, row)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def clear_input_acc(self, layer: int, row: int):
        keys = [k for k in self.input_acc if k[0] == layer and (row is None or k[2] == row)]
        for k in keys:
            self.input_acc.pop(k, None)

    def reset_fired(self, layer: int):
        """
        Call at start of each timestep if model resets firing state.
        """
        keys = [k for k in self.fired if k[0] == layer]
        for k in keys:
            self.fired.pop(k, None)

    # --------------------------------------------------
    # Read APIs (for parity / debugging)
    # --------------------------------------------------
    def get(self, layer: int, ch: int, row: int, col: int) -> float:
        return float(self.vmem.get(self._key(layer, ch, row, col), 0.0))

    def get_fired(self, layer: int, ch: int, row: int, col: int) -> int:
        return int(self.fired.pop(self._key(layer, ch, row, col), 0))

    def items(self) -> Iterator[Tuple[Tuple[int, int, int, int], float]]:
        for k, v in self.vmem.items():
            yield k, v

    def count(self) -> int:
        return len(self.vmem)

    def reset_all(self):
        self.vmem.clear()
        self.fired.clear()
        self.input_acc.clear()

    def is_row_dead(self, layer: int, row: int) -> bool:
        """
        A row is dead if there are no active (unmasked) neurons
        in vmem or input_acc for that (layer,row).
        """
        layer = int(layer)
        row = int(row)

        # Check membrane potentials
        for (L, ch, r, c) in self.vmem.keys():
            if L == layer and r == row:
                if not self.masked.get((L, ch, r, c), 0):
                    return False

        # Check accumulated inputs
        for (L, ch, r, c) in self.input_acc.keys():
            if L == layer and r == row:
                if not self.masked.get((L, ch, r, c), 0):
                    return False

        return True

    def total_active(self) -> int:
        """
        Returns total number of active neuron states
        (vmem + input accumulator).
        """
        return len(self.vmem) + len(self.input_acc)
    
    def kill_row(self, layer, row):
        """
        Remove all neurons belonging to (layer, row)
        for conv-like neurons with keys (L, ch, r, c)manual_tinker.py
        """
        to_delete = []

        for (L, ch, r, c) in self.vmem.keys():
            if L == layer and r == row:
                to_delete.append((L, ch, r, c))

        for key in to_delete:
            self.vmem.pop(key, None)
            self.fired.pop(key, None)
