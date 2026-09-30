from problem import Problem
from config import R_R, X_R, R_L, X_L, S_IT as S0

p = Problem(R_R, X_R, R_L, X_L)
SNR0 = p.SNR_theorem(show=False)
print("Current S_IT =", S0)
print("Current SNR_theorem =", SNR0)

# target around 7 dB (center of 4-10)
SNR_target = 7.0
scale = 10**((SNR_target - SNR0)/10)
print("Scale factor =", scale)
print("New S_IT =", S0*scale)