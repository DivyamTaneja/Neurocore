def compute_receptive_field_rows(layer_shapes, conv_params, pool_params):
    """
    Compute which rows in previous layer affect each row in current layer.

    layer_shapes: dict of layer -> shape tuple (C,H,W) or (N,) for FC
    conv_params: dict of layer -> (K,S,P)  (kernel, stride, padding)
    pool_params: dict of layer -> (Kp, Sp) (kernel, stride)
    """

    num_layers = len(layer_shapes)
    rf_rows = {}

    for L in range(1, num_layers):  # skip input
        shape = layer_shapes[L]
        prev_shape = layer_shapes[L-1]

        # Determine number of rows in current layer
        if len(shape) == 3:
            H_out = shape[1]
        else:
            H_out = 1  # FC layer treated as single "row"

        rf_rows[L] = {}
        for r_out in range(H_out):

            # Fully-connected layer: depends on all previous neurons
            if len(shape) == 1:
                rf_rows[L][r_out] = list(range(prev_shape[0] if len(prev_shape) == 1 else prev_shape[1]))
                continue

            # Convolutional layer
            r_prev_start = 0
            r_prev_end = prev_shape[1]-1 if len(prev_shape) == 3 else 0

            if L in conv_params:
                K, S, P = conv_params[L]
                r_prev_start = r_out * S - P
                r_prev_end = r_prev_start + K - 1

            if L in pool_params:
                Kp, Sp = pool_params[L]
                r_prev_start = r_prev_start * Sp
                r_prev_end = r_prev_end * Sp + (Kp-1)

            # Clamp to valid row indices
            if len(prev_shape) == 3:
                r_prev_start = max(r_prev_start, 0)
                r_prev_end = min(r_prev_end, prev_shape[1]-1)
            else:
                r_prev_start = 0
                r_prev_end = prev_shape[0]-1

            rf_rows[L][r_out] = list(range(r_prev_start, r_prev_end+1))

    return rf_rows


# -----------------------------
# Example usage (same as before)
# -----------------------------
layer_shapes = {
    0: (3, 32, 32),
    1: (32, 16, 16),
    2: (64, 8, 8),
    3: (256,),
    4: (10,)
}

conv_params = {1: (3,1,1), 2: (3,1,1)}
pool_params = {1: (2,2), 2: (2,2)}

rf_rows = compute_receptive_field_rows(layer_shapes, conv_params, pool_params)

for L in rf_rows:
    for r_out, input_rows in rf_rows[L].items():
        print(f"Layer {L} row {r_out} affected by Layer {L-1} rows {input_rows}")
