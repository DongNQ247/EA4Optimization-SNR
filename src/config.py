# Range limits for various electrical parameters
R_R = [1, 1000]
X_R = [-1000, 1000]
R_L = [1, 1000]
X_L = [-1000, 1000]

## Config
g = 1.0             # Hệ số khuếch đại (linear). Tham số hữu ích: 0.1 .. 100
Z_RT = 50 + 0j      # Trở kháng tương thích (50 Ω chuẩn RF)
# Mật độ công suất tín hiệu tại đầu vào (W/Hz)
# Đã được hiệu chỉnh để SNR_theorem ≈ 7 dB (trong khoảng 4–10 dB mong muốn)
S_IT = 8e-23

# Hằng số và tham số nhiễu
k = 1.380649e-23   # Hằng số Boltzmann
T = 290            # Nhiệt độ phòng (K)

# Thông số mặc định để tính nhiễu nội tại (có thể điều chỉnh theo ứng dụng)
# B: băng thông (Hz), NF_dB: noise figure (dB)
B = 1e6            # băng thông mặc định 1 MHz
NF_dB = 3.0        # noise figure mặc định (dB)
F = 10 ** (NF_dB / 10)  # hệ số nhiễu (linear)

# Nhiễu nội tại của bộ thu (W) tính theo N = k * T * B * F
N_na = k * T * B * F

def dbm_per_hz_to_w(dbm_hz):
	"""Chuyển dBm/Hz -> W/Hz"""
	return 10 ** ((dbm_hz - 30) / 10)
