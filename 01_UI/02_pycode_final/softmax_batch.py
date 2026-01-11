import serial
import time
import numpy as np

Q = 10
SCALE = 1 << Q
I16_MIN, I16_MAX = -32768, 32767

BYTES_PER_ROW = 129
MAX_DEPTH = 127


def open_serial(port: str, baud: int = 115200, timeout: float = 1.0) -> serial.Serial:
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=timeout,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def close_serial(ser: serial.Serial) -> None:
    if ser and ser.is_open:
        ser.close()


def send_frame(ser: serial.Serial, depth: int, frames: list[bytes]) -> None:
    if not ser.is_open:
        raise ConnectionError("Serial port is not open.")
    if not (0 <= depth <= MAX_DEPTH):
        raise ValueError(f"depth must be 0..{MAX_DEPTH} (N=depth+1 rows)")

    n_rows = depth + 1
    if len(frames) != n_rows:
        raise ValueError(f"frames length must be depth+1={n_rows}, got {len(frames)}")

    for i, fr in enumerate(frames):
        if not isinstance(fr, (bytes, bytearray)):
            raise TypeError(f"frames[{i}] must be bytes-like, got {type(fr)}")
        if len(fr) != BYTES_PER_ROW:
            raise ValueError(
                f"frames[{i}] must be {BYTES_PER_ROW} bytes, got {len(fr)}"
            )
    ser.reset_input_buffer()

    ser.write(bytes([depth]))
    ser.flush()
    for fr in frames:
        ser.write(fr)
    ser.flush()


def read_exact(ser: serial.Serial, n: int, *, timeout_s: float = 5.0) -> bytes:
    buf = bytearray()
    t0 = time.time()
    while len(buf) < n:
        if (time.time() - t0) > timeout_s:
            raise TimeoutError(f"read_exact timeout: got {len(buf)}/{n} bytes")
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)
    return bytes(buf)


def recv_frames(
    ser: serial.Serial, depth: int, *, timeout_s: float = 10.0
) -> list[bytes]:
    if not (0 <= depth <= MAX_DEPTH):
        raise ValueError(f"depth must be 0..{MAX_DEPTH}")

    n_rows = depth + 1
    total = n_rows * BYTES_PER_ROW
    rx = read_exact(ser, total, timeout_s=timeout_s)
    return [rx[i * BYTES_PER_ROW : (i + 1) * BYTES_PER_ROW] for i in range(n_rows)]


def pack_params(token_len: int) -> tuple[int, int]:
    if not (1 <= token_len <= 64):
        raise ValueError("Length must be between 1 and 64 for pack_params().")
    if token_len <= 16:
        return 16, 4
    elif token_len <= 32:
        return 32, 2
    else:
        return 64, 1


def length_mode(token_len: int) -> int:
    if not (1 <= token_len <= 768):
        raise ValueError("Length must be between 1 and 768.")
    if token_len <= 16:
        return 0
    elif token_len <= 32:
        return 1
    elif token_len <= 64:
        return 2
    elif token_len <= 128:
        return 3
    elif token_len <= 192:
        return 4
    elif token_len <= 256:
        return 5
    elif token_len <= 320:
        return 6
    elif token_len <= 384:
        return 7
    elif token_len <= 448:
        return 8
    elif token_len <= 512:
        return 9
    elif token_len <= 576:
        return 10
    elif token_len <= 640:
        return 11
    elif token_len <= 704:
        return 12
    else:
        return 13


def split_depths(
    total_rows: int, len_mode: int, max_rows_per_tx: int = 128
) -> list[int]:
    if total_rows <= 0:
        return []
    if not (0 <= len_mode <= 15):
        raise ValueError("len_mode must be 0..15")
    if max_rows_per_tx < 1 or max_rows_per_tx > 128:
        raise ValueError("max_rows_per_tx must be 1..128")
    if len_mode <= 2:
        group = 1
    else:
        group = len_mode - 1

    if group > max_rows_per_tx:
        raise ValueError(f"group({group}) > max_rows_per_tx({max_rows_per_tx})")
    if total_rows % group != 0:
        raise ValueError(
            f"total_rows({total_rows}) must be a multiple of group({group}) for len_mode={len_mode}"
        )

    depths: list[int] = []
    rem = total_rows
    while rem > 0:
        groups_fit = min(rem // group, max_rows_per_tx // group)
        rows = groups_fit * group
        depths.append(rows - 1)
        rem -= rows

    return depths


def floats64_to_row_bytes(payload64_f32: np.ndarray, *, header_mode: int) -> bytes:
    x = np.asarray(payload64_f32, dtype=np.float64)
    if x.shape != (64,):
        raise ValueError("payload must be shape (64,)")
    x = np.nan_to_num(x, nan=0.0)
    scaled_f = x * SCALE
    scaled_f = np.clip(scaled_f, I16_MIN, I16_MAX)
    payload_int16 = np.rint(scaled_f).astype(np.int16)
    payload_bytes = payload_int16.astype(">i2", copy=False).tobytes()

    if len(payload_bytes) != 128:
        raise RuntimeError(f"payload_bytes must be 128, got {len(payload_bytes)}")

    header = header_mode & 0x0F
    return bytes([header]) + payload_bytes


def row_bytes_to_floats64(row129: bytes) -> np.ndarray:
    if len(row129) != 129:
        raise ValueError("row must be 129 bytes")
    payload = row129[1:]
    i16 = np.frombuffer(payload, dtype=">i2")
    return i16.astype(np.float64) / SCALE


def softmax_batch(
    ser: serial.Serial,
    scores_list: list[np.ndarray],
    pad_value: float = -32.0,
    timeout_s: float = 10.0,
) -> list[np.ndarray]:
    if not scores_list:
        return []

    seqs = [np.asarray(s, dtype=np.float32).reshape(-1) for s in scores_list]
    L = int(seqs[0].shape[0])
    if not (1 <= L <= 768):
        raise ValueError("Length must be between 1 and 768.")
    if any(int(s.shape[0]) != L for s in seqs):
        raise ValueError(f"All sequences must have the same length {L}.")

    len_mode = length_mode(L)

    frame_list: list[np.ndarray] = []

    if len_mode in (0, 1, 2):
        block_size, pack = pack_params(L)

        for off in range(0, len(seqs), pack):
            chunk = seqs[off : off + pack]
            payload64 = np.full((64,), pad_value, dtype=np.float32)

            for g, vec in enumerate(chunk):
                mini = np.full((block_size,), pad_value, dtype=np.float32)
                mini[:L] = vec
                start = g * block_size
                payload64[start : start + block_size] = mini

            frame_list.append(payload64)

    else:
        for vec in seqs:
            for s in range(0, L, 64):
                payload64 = np.full((64,), pad_value, dtype=np.float32)
                chunk = vec[s : s + 64]
                payload64[: chunk.shape[0]] = chunk
                frame_list.append(payload64)

    frame_bytes_list = [
        floats64_to_row_bytes(p, header_mode=len_mode) for p in frame_list
    ]

    total_rows = len(frame_bytes_list)
    depth_list = split_depths(total_rows, len_mode, max_rows_per_tx=128)

    result_rows: list[bytes] = []
    cursor = 0
    for depth in depth_list:
        n_rows = depth + 1
        frames_to_send = frame_bytes_list[cursor : cursor + n_rows]
        cursor += n_rows

        send_frame(ser, depth, frames_to_send)
        recv_rows = recv_frames(ser, depth, timeout_s=timeout_s)
        result_rows.extend(recv_rows)

    if len(result_rows) != total_rows:
        raise RuntimeError(
            f"RX rows mismatch: got {len(result_rows)}, expected {total_rows}"
        )

    rx_payloads64 = [row_bytes_to_floats64(rb) for rb in result_rows]

    results: list[np.ndarray] = []

    if len_mode in (0, 1, 2):
        block_size, pack = pack_params(L)

        for row_idx, probs64 in enumerate(rx_payloads64):
            off = row_idx * pack
            if off >= len(seqs):
                break
            G = min(pack, len(seqs) - off)

            for g in range(G):
                start = g * block_size
                results.append(np.asarray(probs64[start : start + L], dtype=np.float64))

    else:
        rows_per_softmax = (L + 63) // 64
        idx = 0
        for _ in range(len(seqs)):
            concat = np.concatenate(rx_payloads64[idx : idx + rows_per_softmax], axis=0)
            results.append(concat[:L].astype(np.float64, copy=False))
            idx += rows_per_softmax

    if len(results) != len(seqs):
        raise RuntimeError(
            f"result count mismatch: got {len(results)}, expected {len(seqs)}"
        )

    return results
