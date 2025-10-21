import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
# -------------------------------
# 1. True vehicle parameters (same as before)
# -------------------------------
Iz_true = 1800.0   # kg·m²
a, b = 1.2, 1.6    # m
Cf, Cr = 60000.0, 60000.0  # N/rad
u = 20.0            # m/s (constant speed)

rng = np.random.default_rng(42)

# -------------------------------
# 2. Generate improved excitation data
# -------------------------------
t = np.linspace(0, 40, 4000)  # 40 s, 100 Hz

# Multi-frequency steering input
delta = 0.02*np.sin(0.5*t) + 0.01*np.sin(1.5*t + 0.8) + 0.005*np.sin(2.3*t + 1.2)
# Small noise to simulate variability
delta += rng.normal(0, 0.0005, size=len(t))

# Simulate vehicle yaw rate & lateral velocity responses (phase shifted)
r = 0.04*np.sin(0.5*t - 0.3) + 0.02*np.sin(1.5*t - 0.5) + 0.01*np.sin(2.3*t - 0.8)
vy = 0.3*np.sin(0.5*t - 0.6) + 0.15*np.sin(1.5*t - 0.9) + 0.05*np.sin(2.3*t - 1.1)

# Compute yaw acceleration using true model
r_dot = (a*Cf/Iz_true)*delta \
        - ((a**2*Cf + b**2*Cr)/(Iz_true*u))*r \
        - ((a*Cf - b*Cr)/(Iz_true*u))*vy

# Add small measurement noise
r_dot_meas = r_dot + rng.normal(0, 0.002, size=len(t))
"""

# Open and read file
LapFive = pd.read_csv("Matt_Romanowski_Watkins_Glenn_2022_lap5.csv")

headers = LapFive.columns

steer = LapFive["STEERINGPOSITION"].values
yaw = LapFive["YawRate"].values
LatAcc = LapFive["LateralAcc"].values

# -------------------------------
# 3. Build regression matrix
# -------------------------------
Phi = np.column_stack((steer, yaw, LatAcc))
Y = r_dot_meas

# -------------------------------
# 4. SVD diagnostics
# -------------------------------
U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
print("Singular values of Phi:\n", S)
print("Condition number (sigma1 / sigma_n):", S[0] / S[-1])

# naive inversion (what we did earlier)
theta_svd_naive = Vt.T @ np.diag(1.0 / S) @ U.T @ Y
print("\nTheta (naive SVD inversion):", theta_svd_naive)
A1 = theta_svd_naive[0]
print("A1 (naive) =", A1)
if abs(A1) < 1e-12:
    print("A1 is essentially zero -> leads to Iz ~ inf or Iz ~ 0 when dividing. Bad/unstable.")

# -------------------------------
# 5. More stable: use pseudoinverse (np.linalg.pinv)
# -------------------------------
theta_pinv = np.linalg.pinv(Phi) @ Y
print("\nTheta (np.linalg.pinv):", theta_pinv)
A1_pinv = theta_pinv[0]
Iz_est_pinv = a * Cf / A1_pinv if abs(A1_pinv) > 1e-12 else np.nan
print("Estimated I_z (pinv) = ", Iz_est_pinv)

# -------------------------------
# 6. Truncated SVD (regularization) - drop tiny singular values
# -------------------------------
tol = 1e-6 * S[0]  # relative tolerance, adjust if needed
large_idx = S > tol
print("\nTruncation tolerance:", tol, " - keeping", large_idx.sum(), "singular values out of", len(S))
S_inv_trunc = np.zeros_like(S)
S_inv_trunc[large_idx] = 1.0 / S[large_idx]
theta_trunc = Vt.T @ np.diag(S_inv_trunc) @ U.T @ Y
print("Theta (truncated SVD):", theta_trunc)
A1_trunc = theta_trunc[0]
Iz_est_trunc = a * Cf / A1_trunc if abs(A1_trunc) > 1e-12 else np.nan
print("Estimated I_z (trunc SVD) = ", Iz_est_trunc)

# -------------------------------
# 7. Quick comparison plot
# -------------------------------
r_dot_pred_pinv = Phi @ theta_pinv

plt.figure(figsize=(8, 4))
plt.plot(t, r_dot_meas, label='Measured yaw acceleration', alpha=0.6)
plt.plot(t, r_dot_pred_pinv, '--', label='Model prediction (pinv fit)', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Yaw acceleration [rad/s²]')
plt.legend()
plt.title('Yaw Acceleration Fit using pinv (stable)')
plt.grid(True)
plt.show()
