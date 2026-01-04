import numpy as np
import serial

# 새로 만든 가변 길이 softmax(UART 누적 버퍼 방식)
# softmax_packet.py 안에 softmax_fpga_variable()이 있다고 가정
from softmax_packet import softmax_fpga_variable


def softmax_FPGA_UART_batch(
    ser: serial.Serial,
    scores_list,
    *,
    pad_value: float = -32.0,
    deadline_s: float = 2.0,
):
    """
    scores_list: [vec0, vec1, ...] where each vec is length L (1..768)
    return: [prob0, prob1, ...] each length L

    NOTE:
    - 기존 코드의 '패킹(max_pack)'은 새 FPGA 프로토콜(누적 버퍼 + 1회 softmax)과 맞지 않아서 제거.
    - 각 vec마다 softmax_fpga_variable()을 독립 호출.
    """
    seqs = [np.asarray(s, dtype=np.float64).reshape(-1) for s in scores_list]
    if not seqs:
        return []

    L = int(seqs[0].shape[0])
    if any(int(s.shape[0]) != L for s in seqs):
        raise ValueError("All sequences must have the same length.")
    if not (1 <= L <= 768):
        raise ValueError("Length must be between 1 and 768.")

    results = []
    for vec in seqs:
        probs = softmax_fpga_variable(
            ser,
            vec,
            pad_value=pad_value,
            deadline_s=deadline_s,
        )
        results.append(np.asarray(probs, dtype=np.float64))
    return results


def attention(
    Q,
    K,
    V,
    ser: serial.Serial,
    *,
    pad_value: float = -32.0,
    deadline_s: float = 2.0,
    return_attn: bool = False,
):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    Nq, d_kq = Q.shape
    Nk, d_kk = K.shape
    Nv, d_kv = V.shape

    assert Nq == Nk and d_kq == d_kk, "Dim Error: Q,K 크기 불일치"
    assert Nv == Nk, "Dim Error: V의 행 수는 K의 행 수와 같아야 합니다."
    assert 1 <= Nk <= 768, "Length N must be between 1 and 768."

    N = Nk
    d_k = d_kq

    outputs = np.zeros((Nq, d_kv), dtype=np.float64)
    attn_mat = np.zeros((N, N), dtype=np.float64) if return_attn else None

    for i in range(Nq):
        scores = (K @ Q[i].T) / np.sqrt(d_k)  # (N,)
        probs = softmax_fpga_variable(  # (N,)
            ser,
            scores,
            pad_value=pad_value,
            deadline_s=deadline_s,
        )

        if return_attn:
            attn_mat[i, :] = probs

        outputs[i, :] = np.asarray(probs, dtype=np.float64) @ V

    if return_attn:
        return outputs, attn_mat
    return outputs
