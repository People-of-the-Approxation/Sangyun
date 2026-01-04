import math
import numpy as np

from UART_base import (
    floats_to_q610_bytes,
    q610_bytes_to_floats,
    send_exact,
    read_exact,
)


def length_to_tag_and_padlen(L: int) -> tuple[int, int]:
    """
    tag mapping:
      0:16, 1:32, 2:64, 3:128, 4:192, 5:256, ... 13:768
    """
    L = int(L)
    if L < 1:
        L = 1
    if L > 768:
        L = 768

    if L <= 128:
        tag = max(0, int(math.ceil(math.log2(L))) - 4)
        P = 16 << tag
        return tag, P

    tag = 3 + int(math.ceil((L - 128) / 64.0))
    tag = min(tag, 13)
    P = 128 + 64 * (tag - 3)
    return tag, P


def build_softmax_frame_chunk64(
    chunk64: np.ndarray, tag: int, *, endian: str = "little"
) -> bytes:
    """
    TX frame: 129B = [tag 1B] + [payload 128B] where payload is 64 int16(Q6.10)
    """
    x = np.asarray(chunk64, dtype=np.float64).reshape(-1)
    if x.size != 64:
        raise ValueError(f"chunk must be length 64, got {x.size}")
    payload_bytes = floats_to_q610_bytes(x, endian=endian)
    return bytes([tag & 0xFF]) + payload_bytes


def softmax_fpga_variable(
    ser,
    vec: np.ndarray,
    *,
    pad_value: float = -32.0,
    endian: str = "little",
    deadline_s: float = 2.0,
) -> np.ndarray:
    """
    Host→FPGA:
      repeat P/64 times:
        send 129B (tag + 64 scores)
    FPGA→Host:
      repeat P/64 times:
        read 128B (64 probs)
    return probs[:L]
    """
    scores = np.asarray(vec, dtype=np.float64).reshape(-1)
    L = int(scores.size)
    if not (1 <= L <= 768):
        raise ValueError("Length must be between 1 and 768.")

    tag, P = length_to_tag_and_padlen(L)

    # RX is always 128B per chunk by your design
    # For small P (16/32), you still want 1 chunk RX(128B). 통일 규칙이면 아래가 맞음.
    if P < 64:
        P_eff = 64
        nframes = 1
    else:
        P_eff = P
        nframes = P_eff // 64

    padded = np.full(P_eff, pad_value, dtype=np.float64)
    padded[:L] = scores

    # TX
    for k in range(nframes):
        chunk = padded[k * 64 : (k + 1) * 64]
        frame = build_softmax_frame_chunk64(chunk, tag=tag, endian=endian)
        send_exact(ser, frame)

    # RX
    probs = np.empty(P_eff, dtype=np.float64)
    for k in range(nframes):
        buf = read_exact(ser, 128, deadline_s=deadline_s)
        probs[k * 64 : (k + 1) * 64] = q610_bytes_to_floats(buf, endian=endian)

    return probs[:L]
