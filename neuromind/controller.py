# controller.py

# ------------------------------------------------------------
# Receptive field computation (GROUND TRUTH)
# ------------------------------------------------------------
import numpy as np

def compute_receptive_field_rows(layer_shapes, conv_params, pool_params):
    """
    Compute which rows in previous layer affect each row in current layer.

    layer_shapes: dict[layer] -> shape tuple
                  (C,H,W) for conv / input, (N,) for FC
    conv_params: dict[layer] -> (K, S, P)
    pool_params: dict[layer] -> (Kp, Sp)
    """

    rf_rows = {}

    layers = sorted(layer_shapes.keys())

    for idx in range(1, len(layers)):
        L = layers[idx]
        Lprev = layers[idx - 1]

        shape = layer_shapes[L]
        prev_shape = layer_shapes[Lprev]

        # number of rows in current layer
        if len(shape) == 3:
            H_out = shape[1]
        else:
            H_out = 1

        rf_rows[L] = {}

        for r_out in range(H_out):

            # -------- Fully connected --------
            if len(shape) == 1:
                if len(prev_shape) == 3:
                    rf_rows[L][r_out] = list(range(prev_shape[1]))
                else:
                    rf_rows[L][r_out] = [0]
                continue

            # -------- Conv --------
            Kc, Sc, Pc = conv_params[L]
            r0 = r_out * Sc - Pc
            r1 = r0 + Kc - 1

            # -------- Pool --------
            if L in pool_params:
                Kp, Sp = pool_params[L]
                r0 = r0 * Sp
                r1 = r1 * Sp + (Kp - 1)

            # Clamp
            r0 = max(0, r0)
            r1 = min(prev_shape[1] - 1, r1)

            rf_rows[L][r_out] = list(range(r0, r1 + 1))

    return rf_rows


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def num_rows(shape):
    """
    Conv/Input: (C,H,W) -> H
    FC: (N,) -> 1
    """
    if len(shape) == 3:
        return shape[1]
    elif len(shape) == 1:
        return 1
    else:
        raise ValueError(f"Unknown shape {shape}")


# ------------------------------------------------------------
# Controller state
# ------------------------------------------------------------
class ControllerState:
    def __init__(self, processor, Tmax):
        self.processor = processor
        self.Tmax = Tmax

        # --------------------------------------------------
        # Execution time (USED BY SCHEDULER)
        # --------------------------------------------------
        self.exec_t = {}
        self.commit_t = {}
        self.row_release_t = {}

        for L, shape in processor.shapes.items():
            for r in range(num_rows(shape)):
                self.exec_t[(L, r)] = 0
                self.commit_t[(L, r)] = 0
                self.row_release_t[(L, r)] = 0

        # --------------------------------------------------
        # Precompute receptive fields
        # --------------------------------------------------
        self.rf_rows = compute_receptive_field_rows(
            layer_shapes=processor.shapes,
            conv_params=processor.conv_params,
            pool_params=processor.pool_params
        )

    def reset(self):
        """
        Reset dynamic execution state for a new episode
        """
        self.exec_t.clear()
        self.commit_t.clear()

        for L, shape in self.processor.shapes.items():
            for r in range(num_rows(shape)):
                self.exec_t[(L, r)] = 0
                self.commit_t[(L, r)] = 0

# ------------------------------------------------------------
# Action space (STRICT dependency check)
# ------------------------------------------------------------
def get_action_space(state, processor):
    actions = []

    for L, shape in processor.shapes.items():
        H = num_rows(shape)

        for R in range(H):
            t_cur = state.exec_t[(L, R)]

            if t_cur > state.row_release_t[(L, R)]:
                continue

            # timestep exhausted
            if t_cur >= state.Tmax:
                continue

            # input layer always schedulable
            if L == 0:
                actions.append((L, R))
                continue

            deps = state.rf_rows[L][R]

            # ✅ STRICT causality: RF input must be ahead
            ready = all(
                state.exec_t[(L - 1, rin)] > t_cur
                for rin in deps
            )


            if ready:
                actions.append((L, R))

    return actions


# ------------------------------------------------------------
# Execute exactly one scheduling step
# ------------------------------------------------------------
def execute_action(action, state, processor, spike_mem, neuron_mem):
    L, R = action
    t_exec = state.exec_t[(L, R)]

    # --------------------------------------------------
    # 1. Execute row (SCHEDULING CLOCK)
    # --------------------------------------------------
    row_spikes = spike_mem.get_spikes(L, t_exec, R)
    processor.process_row_spikes(L, R, row_spikes)

    state.exec_t[(L, R)] += 1

    # --------------------------------------------------
    # 2. RF-driven commits (COMMIT CLOCK)
    # --------------------------------------------------
    next_layer = L + 1
    if next_layer not in state.rf_rows:
        return

    for r_out, rf_in_rows in state.rf_rows[next_layer].items():

        ready = all(
            state.exec_t[(L, rin)] > state.commit_t[(next_layer, r_out)]
            for rin in rf_in_rows
        )

        if ready:
            t_out = state.commit_t[(next_layer, r_out)]
            processor.commit_layer(next_layer, r_out, t_out)
            print("Commited!", "Layer - ", next_layer, "Row - ", r_out, "Time - ", t_out)
            state.commit_t[(next_layer, r_out)] += 1

            for rin in rf_in_rows:
                state.row_release_t[(L, rin)] = max(
                    state.row_release_t[(L, rin)],
                    t_out + 1
                )

        # --------------------------------------------------
        # 3. RF closure → kill row
        # --------------------------------------------------
        rf_closed = all(
            state.exec_t[(L, rin)] >= state.Tmax
            for rin in rf_in_rows
        )

        if rf_closed:
            neuron_mem.kill_row(next_layer, r_out)


def get_observation(spike_mem, neuron_mem, state, actions):
    exec_ts = list(state.exec_t.values()) if state.exec_t else [0]
    return np.array([
        spike_mem.count_total_spikes(),
        neuron_mem.total_active(),
        len(actions),
        min(exec_ts),
        max(exec_ts),
    ], dtype=np.float32)

def compute_reward(
    prev_spikes,
    prev_neurons,
    spike_mem,
    neuron_mem,
    done,
    deadlock,
    *,
    action_repeated=False
):
    """
    Reward shaping for spike scheduling RL

    Goals:
    - Reduce spike memory
    - Reduce active neuron memory
    - Penalize no-progress actions
    - Penalize repetition
    - Encourage fast termination
    """

    reward = 0.0

    # -------------------------------------------------
    # Current state
    # -------------------------------------------------
    cur_spikes = spike_mem.count_total_spikes()
    cur_neurons = neuron_mem.total_active()

    d_spikes = prev_spikes - cur_spikes
    d_neurons = prev_neurons - cur_neurons

    # -------------------------------------------------
    # 1. Positive progress reward
    # -------------------------------------------------
    reward += 1.0 * d_spikes
    reward += 0.5 * d_neurons

    # -------------------------------------------------
    # 2. STRONG penalty for no progress
    # -------------------------------------------------
    if d_spikes <= 0 and d_neurons <= 0:
        reward -= 5.0   # 🔥 kills "safe action" collapse

    # -------------------------------------------------
    # 3. Time penalty (urgency)
    # -------------------------------------------------
    reward -= 1.0       # 🔥 MUST be >= 1.0 for scheduling

    # -------------------------------------------------
    # 4. Repeated action penalty
    # -------------------------------------------------
    if action_repeated:
        reward -= 2.0

    # -------------------------------------------------
    # 5. Terminal rewards
    # -------------------------------------------------
    if done:
        reward += 100.0   # success
    if deadlock:
        reward -= 100.0   # failure

    return reward

# ------------------------------------------------------------
# Simple controller loop (baseline)
# ------------------------------------------------------------
def run_controller(processor, spike_mem, neuron_mem, Tmax):
    state = ControllerState(processor, Tmax)

    output_layer = max(processor.shapes.keys())
    step = 0

    while True:

        if spike_mem.has_output_spike(output_layer):
            print(f"[Controller] Output spike at step {step}")
            break

        actions = get_action_space(state, processor, neuron_mem)

        if not actions:
            print("[Controller] Deadlock")
            break

        # baseline policy: deeper layer first
        actions.sort(key=lambda x: (-x[0], x[1]))
        action = actions[0]

        execute_action(action, state, processor, spike_mem, neuron_mem)
        step += 1
