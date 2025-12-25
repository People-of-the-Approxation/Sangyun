import numpy as np
import matplotlib.pyplot as plt
import math

# 1. Simulation Setup
frac_bits_in = 11  # Input: UQ5.11
frac_bits_out = 11  # Output: S1.4.11
shift_amt = 8  # Slope Shift Amount (>> 8)

# Input Range: 0 ~ 32
x = np.arange(1, 65536) / (2**frac_bits_in)

# Ground Truth (Ideal Math)
y_real = 1.0 / np.sqrt(x)

coeff_table = [
    (15, -1, 497),
    (14, -3, 703),
    (13, -9, 994),
    (12, -26, 1406),
    (11, -73, 1988),
    (10, -206, 2812),
    (9, -583, 3977),
    (8, -1648, 5624),
    (7, -4662, 7954),
    (6, -13185, 11249),
    (5, -37294, 15908),
    (4, -105483, 22497),
    (3, -298352, 31816),
    (2, -843868, 44995),
    (1, -2386819, 63632),
    (0, -6750944, 89989),
]

y_hw = []

# Bit-exact Simulation Loop
for val in x:
    if val < (2**-11):
        seg_idx = 0
    else:
        exponent = math.floor(math.log2(val))
        bit_pos = exponent + frac_bits_in
        seg_idx = max(0, min(15, bit_pos))

    slope = 0
    base = 0
    for row in coeff_table:
        if row[0] == seg_idx:
            slope = row[1]
            base = row[2]
            break

    seg_start_val = 2 ** math.floor(math.log2(val)) if seg_idx != 0 else (2**-11)
    mantissa_int = math.floor((val - seg_start_val) * (2**frac_bits_in))

    mul_res = slope * mantissa_int
    shifted_res = mul_res >> shift_amt
    hw_result_int = shifted_res + base

    if hw_result_int > 32767:
        hw_result_int = 32767
    y_hw.append(hw_result_int / (2**frac_bits_out))

y_hw = np.array(y_hw)

# 3. Error Calculation
abs_error_val = y_hw - y_real
rel_error_val = (abs_error_val / (y_real + np.finfo(float).eps)) * 100

# 4. Plotting
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

# [Plot 1] Function Comparison
ax1 = axes[0]
ax1.plot(x, y_real, color="black", linestyle="-", linewidth=2.0, label="Ideal Math")
ax1.plot(x, y_hw, color="red", linestyle="--", linewidth=1.5, label="Hardware (UQ5.11)")

ax1.set_title(
    "1. Function Approximation ($y = 1/\sqrt{v}$)", fontsize=14, fontweight="bold"
)
ax1.set_ylabel("Output Value", fontsize=12)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.set_xlim(0, 32)
ax1.set_ylim(0, 10)

# [Plot 2] Absolute Error
ax2 = axes[1]
ax2.plot(x, abs_error_val, color="magenta", linewidth=1.0)
ax2.axhline(0, color="black", linewidth=0.5)

ax2.set_title("2. Absolute Error ($y_{hw} - y_{real}$)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Error Value", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.set_xlim(0, 32)
ax2.set_ylim(-0.2, 0.2)

# [Plot 3] Relative Error
ax3 = axes[2]
ax3.plot(x, rel_error_val, color="blue", linewidth=1.0)
ax3.axhline(0, color="black", linewidth=0.5)

ax3.set_title("3. Relative Error (%)", fontsize=14, fontweight="bold")
ax3.set_xlabel(r"Input Variance ($v = \sigma^2$)", fontsize=12)
ax3.set_ylabel("Error (%)", fontsize=12)
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.set_xlim(0, 32)
ax3.set_ylim(-5, 5)

save_filename = "checking the error and graph.png"
plt.savefig(save_filename, dpi=300)
print(f"Graph saved to {save_filename}")

plt.show()
