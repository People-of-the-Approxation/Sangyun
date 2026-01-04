import serial
import time
import numpy as np

Q = 10
SCALE = 1 << Q
I16_MIN, I16_MAX = -32768, 32767
Q610_MIN, Q610_MAX = -32.0, (32.0 - 1.0 / SCALE)


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

    # ESP32/FTDI/Arduino 계열은 포트 오픈 후 리셋/부트 딜레이가 필요할 때가 많음
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def send_exact(ser: serial.Serial, frame: bytes):
    # 기존 코드 스타일 유지: 전송 전 입력 버퍼 비움
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    return


def read_exact(ser: serial.Serial, N: int, deadline_s: float = 2.0) -> bytes:
    end_time_s = time.perf_counter() + deadline_s
    buffer = bytearray()

    while len(buffer) < N:
        chunk = ser.read(N - len(buffer))
        if chunk:
            buffer.extend(chunk)
        else:
            if time.perf_counter() > end_time_s:
                raise TimeoutError(
                    f"Error: timeout -> read_exact() [{len(buffer)}/{N}]"
                )
    return bytes(buffer)


def floats_to_q610_bytes(
    x64, *, endian: str = "little", mode: str = "saturate"
) -> bytes:
    """
    Convert float array to Q6.10 int16 bytes.
    - 기존: 길이 64 고정
    - 수정: 1D 임의 길이 허용(64, 128, 256 ... 또는 16/32 등도 가능)
    """
    x = np.asarray(x64, dtype=np.float64).reshape(-1)

    if x.size < 1:
        raise ValueError("Input must contain at least 1 element.")

    if mode == "strict":
        if np.any(x < Q610_MIN) or np.any(x > Q610_MAX):
            idx = int(np.where((x < Q610_MIN) | (x > Q610_MAX))[0][0])
            raise OverflowError(
                f"Q6.10 range exceeded (index={idx}, value={x[idx]:.6f})"
            )

    scaled = np.rint(x * SCALE).astype(np.int32)

    # saturate가 기본 동작
    if mode in ("saturate", "strict"):
        scaled = np.clip(scaled, I16_MIN, I16_MAX)

    scaled = scaled.astype(np.int16)

    dtype = "<i2" if endian == "little" else ">i2"
    return scaled.astype(dtype, copy=False).tobytes()


def q610_bytes_to_floats(b: bytes, *, endian: str = "little") -> np.ndarray:
    """
    Convert Q6.10 int16 bytes -> float array.
    - 기존: 128B(=64개) 고정
    - 수정: 바이트 길이 기반(2의 배수면 OK)
    """
    if len(b) % 2 != 0:
        raise ValueError(f"Input byte length must be a multiple of 2 (len={len(b)})")

    dtype = "<i2" if endian == "little" else ">i2"
    i16 = np.frombuffer(b, dtype=dtype)
    return i16.astype(np.float64) / SCALE


def build_softmax_frame_chunk64(
    payload64: np.ndarray, tag: int, *, endian: str = "little"
) -> bytes:
    """
    NEW PROTOCOL (TX):
      129B = [tag 1B] + [payload 128B]  (payload = 64x int16(Q6.10))
    """
    x = np.asarray(payload64, dtype=np.float64).reshape(-1)
    if x.size != 64:
        raise ValueError(f"payload64 must have length 64 (got {x.size})")

    payload_bytes = floats_to_q610_bytes(x, endian=endian)
    return bytes([int(tag) & 0xFF]) + payload_bytes


# ---- (optional) legacy helper kept for backward compatibility ----
def build_softmax_frame(
    payload64: np.ndarray, length: int, *, endian: str = "little"
) -> bytes:
    """
    LEGACY PROTOCOL (TX):
      129B = [L 1B] + [payload 128B]
    - 과거에는 L(1..64) 의미로 사용했음.
    - 새 설계에서는 tag 방식(build_softmax_frame_chunk64)을 쓰는 것을 권장.
    """
    L = int(length)
    if L < 1:
        L = 1
    if L > 64:
        L = 64

    x = np.asarray(payload64, dtype=np.float64).reshape(-1)
    if x.size != 64:
        raise ValueError(f"payload64 must have length 64 (got {x.size})")

    payload_bytes = floats_to_q610_bytes(x, endian=endian)
    return bytes([L]) + payload_bytes
