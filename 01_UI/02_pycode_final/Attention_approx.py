import numpy as np
import serial
from softmax_batch import softmax_batch


def attention(
    Q,
    K,
    V,
    ser: serial.Serial,
    *,
    pad_value: float = -32.0,
    timeout_s: float = 10.0,
) -> np.ndarray:

    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V must be 2D arrays")

    Nq, d_kq = Q.shape
    Nk, d_kk = K.shape
    Nv, d_v = V.shape

    if d_kq != d_kk:
        raise ValueError(f"Dim mismatch: Q{Q.shape}, K{K.shape}")
    if Nv != Nk:
        raise ValueError(f"Dim mismatch: V{V.shape}, K{K.shape} (Nv must equal Nk)")

    d_k = d_kq

    S = (K @ Q.T) / np.sqrt(d_k)

    seqs = [S[:, j].astype(np.float32, copy=False) for j in range(Nq)]

    probs_list = softmax_batch(
        ser,
        seqs,
        pad_value=pad_value,
        timeout_s=timeout_s,
    )

    P = np.vstack([np.asarray(p, dtype=np.float64) for p in probs_list])
    if P.shape != (Nq, Nk):
        raise RuntimeError(f"softmax_batch returned {P.shape}, expected {(Nq, Nk)}")

    out = P @ V
    return out
