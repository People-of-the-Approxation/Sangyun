import numpy as np
import serial
from UART_base import build_softmax_frame, send_exact, read_exact, q610_bytes_to_floats


def _pack_params(L: int):
    if not (1 <= L <= 64):
        raise ValueError("Length must be between 1 and 64.")
    if L <= 16:
        return 16, 4
    elif L <= 32:
        return 32, 2
    else:
        return 64, 1


def softmax_FPGA_UART_batch(
    ser: serial.Serial,
    scores_list,
    *,
    pad_value: float = -32.0,
    deadline_s: float = 2.0
):
    seqs = [np.asarray(s, dtype=np.float64) for s in scores_list]
    if not seqs:
        return []
    L = int(seqs[0].shape[0])
    if any(int(s.shape[0]) != L for s in seqs):
        raise ValueError("All sequences must have the same length.")
    block_size, max_pack = _pack_params(L)

    results = []
    for off in range(0, len(seqs), max_pack):
        chunk = seqs[off : off + max_pack]
        G = len(chunk)

        payload = np.full(64, pad_value, dtype=np.float64)
        for g, vec in enumerate(chunk):
            start = g * block_size
            payload[start : start + L] = vec

        frame = build_softmax_frame(payload, length=L, endian="little")
        send_exact(ser, frame)

        probs64 = q610_bytes_to_floats(
            read_exact(ser, 128, deadline_s=deadline_s), endian="little"
        )

        for g in range(G):
            start = g * block_size
            results.append(np.asarray(probs64[start : start + L], dtype=np.float64))

    return results


def attention(
    Q, K, V, ser: serial.Serial, *, pad_value: float = -32.0, deadline_s: float = 2.0
):

    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    Nq, d_kq = Q.shape
    Nk, d_kk = K.shape
    Nv, d_kv = V.shape

    assert Nq == Nk and d_kq == d_kk, "Dim Error: Q,K 크기 불일치"
    assert Nv == Nk, "Dim Error: V의 행 수는 K의 행 수와 같아야 합니다."
    assert 1 <= Nk <= 64, "Length N must be between 1 and 64."

    N = Nk
    d_k = d_kq
    _, max_pack = _pack_params(N)

    outputs = np.zeros((Nq, d_kv), dtype=np.float64)

    for off in range(0, Nq, max_pack):
        Qb = Q[off : off + max_pack]
        B = Qb.shape[0]

        S = (K @ Qb.T) / np.sqrt(d_k)

        seqs = [S[:, j] for j in range(B)]
        probs = softmax_FPGA_UART_batch(
            ser, seqs, pad_value=pad_value, deadline_s=deadline_s
        )
        F = np.vstack(probs)

        outputs[off : off + B, :] = F @ V

    return outputs
