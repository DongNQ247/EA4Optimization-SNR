import numpy as np
from config import R_R, X_R, R_L, X_L, g, Z_RT, S_IT, N_na, T, k
   
## Set
alpha = N_na
beta  = g**2*2*k*T

class Problem:
    def __init__(self, R_R, X_R, R_L, X_L):
        self.R_R = R_R
        self.X_R = X_R
        self.R_L = R_L
        self.X_L = X_L
   
    # Tính SNR theo các công thức đã cho
    def SNR_theorem(self, show=False):
        print("SNR_theorem")

        results = []  # Lưu các case để in bảng

        # CASE LIST: (x_index, y_index)
        cases = [
            (0, 0),    # Case 1
            (0, -1),   # Case 2
            (-1, 0),   # Case 3
            (-1, -1)   # Case 4
        ]

        for i, (ix, iy) in enumerate(cases, start=1):
            x = self.R_L[ix]
            y = self.X_L[iy]

            a = self.R_R[0]
            b = -(alpha * y) / (alpha + beta * x)

            Z_R = complex(a, b)
            Z_L = complex(x, y)

            SNR_val = self.SNR(Z_R, Z_L)

            results.append((f"Case {i}", Z_R, Z_L, SNR_val))

        # Tìm SNR max
        SNR_max = max(r[3] for r in results)

        # In bảng
        if show:
            print(f"{'Case':<6} | {'Z_R':<30} | {'Z_L':<30} | {'SNR':<20}")
            print("-" * 96)
            for case, Z_R, Z_L, SNR_val in results:
                print(f"{case:<6} | {str(Z_R):<30} | {str(Z_L):<30} | {SNR_val:<20}")

            print("-" * 96)
            print(f"{'SNR_max':<6} = {SNR_max}")
            print("-" * 96)
            
        return SNR_max
    
    # Chuyển đổi giá trị sang dB
    def dB(self, value):
        return 10*np.log10(value)
    
    # Tính SNR với Z_R, Z_L cụ thể
    def SNR(self, Z_R=None, Z_L=None):
        numerator   = g**2 * abs(Z_L/(Z_R + Z_L))**2 * abs(Z_RT)**2 * S_IT
        denominator = N_na + g**2 * abs(Z_R/(Z_R + Z_L))**2 * 2 * k * T * np.real(Z_L)
        return self.dB(numerator / denominator) # (dB)
