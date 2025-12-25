import numpy as np

FUNC = lambda x: 1.0 / np.sqrt(x)
TOTAL_BITS = 16
FRAC_BITS = 11      # UQ5.11
OUT_FRAC_BITS = 11  # S1.4.11
SLOPE_SHIFT_AMT = 8

print(f"--- Optimized PWL Coefficients (Least Squares) ---")
print("// Verilog Case Statement")
print("always @(*) begin")
print("    case (segment_idx)")

for bit_idx in range(TOTAL_BITS - 1, -1, -1):
    power_val = bit_idx - FRAC_BITS
    x_start = 2.0**power_val
    x_end = 2.0 ** (power_val + 1)

    x_samples = np.linspace(x_start, x_end, 100)
    y_samples = FUNC(x_samples)

    # 1차원 직선(y = mx + c) 피팅 (Least Squares)
    # m: 기울기, c: 절편
    m, c = np.polyfit(x_samples, y_samples, 1)

    y_base_fitted = m * x_start + c
    base_int = int(round(y_base_fitted * (2**OUT_FRAC_BITS)))

    scale_factor = SLOPE_SHIFT_AMT - FRAC_BITS + OUT_FRAC_BITS
    slope_int = int(round(m * (2**scale_factor)))

    print(
        f"        4'd{bit_idx}: begin slope = {slope_int}; base = {base_int}; end // Range [{x_start} ~ {x_end}]"
    )

print("        default: begin slope = 0; base = 0; end")
print("    endcase")
print("end")
