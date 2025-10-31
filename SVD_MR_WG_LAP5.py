import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# -------------------------------
# Vehicle params (needed for Iz estimate)
# -------------------------------
G = 17.78
g = 9.81
L = 2.45
a = 1.225
b = 1.4
Cf = 60000

# -------------------------------
# Load CSV and show headers
# -------------------------------
fn = "Matt_Romanowski_Watkins_Glenn_2022_lap5.csv"
LapFive = pd.read_csv(fn)
# print("Columns in CSV:\n", list(LapFive.columns))

# -------------------------------
# Find/assign columns
# -------------------------------
yaw_col = "YawRate"
steer_col = "STEERINGPOSITION"
latacc_col = "LateralAcc"

if yaw_col not in LapFive.columns:
    raise KeyError(f"Couldn't find '{yaw_col}' in CSV columns. See printed list above and adjust yaw_col.")

if steer_col not in LapFive.columns:
    warnings.warn(f"Couldn't find '{steer_col}' in CSV columns. Steering will be set to zeros.", UserWarning)
    SWsteerDEG = np.zeros(len(LapFive))
else:
    SWsteerDEG = LapFive[steer_col].values.astype(float)

if latacc_col not in LapFive.columns:
    warnings.warn(f"Couldn't find '{latacc_col}' in CSV columns. Lateral Acc set to zeros.", UserWarning)
    LatAccG = np.zeros(len(LapFive))
else:
    LatAccG = LapFive[latacc_col].values.astype(float)

yawDeg = LapFive[yaw_col].values.astype(float)

# -------------------------------
# Convert units: 
# -------------------------------
LatAccMS = 9.81*LatAccG
FWsteerRad = (SWsteerDEG/G)*np.pi/180
yawRad = yawDeg*np.pi/180

# -------------------------------
# Build time vector: prefer a time column, else assume 100Hz
# -------------------------------
time_candidates = [c for c in ["Time", "time", "Timestamp", "timestamp", "t"] if c in LapFive.columns]
if time_candidates:
    tcol = time_candidates[0]
    t_raw = LapFive[tcol]
    if np.issubdtype(t_raw.dtype, np.number):
        t = t_raw.values.astype(float)
    else:
        # try parse datetimes
        try:
            t_pd = pd.to_datetime(t_raw)
            t = (t_pd - t_pd.iloc[0]).dt.total_seconds().values
        except Exception:
            warnings.warn("Couldn't parse time column; falling back to index/100Hz.")
            t = np.arange(len(yawRad)) / 100.0
else:
    warnings.warn("No time column found; assuming 100 Hz sample rate.")
    t = np.arange(len(yawRad)) / 100.0

# Ensure arrays have same length
N = min(len(t), len(yawRad), len(FWsteerRad), len(LatAccMS))
t = t[:N]; yawRad = yawRad[:N]; FWsteerRad = FWsteerRad[:N]; LatAccMS = LatAccMS[:N]

# -------------------------------
# Compute yaw acceleration r_dot_meas from measured yaw rate
# -------------------------------
# central difference / gradient with smoothing
r_dot_meas = np.gradient(yawRad, t)  # derivative
# simple smoothing to reduce differentiation noise (moving average)
win = 5  # samples, change if too aggressive
if win > 1:
    kernel = np.ones(win) / win
    r_dot_meas = np.convolve(r_dot_meas, kernel, mode='same')

# -------------------------------
# Build regression matrix Phi and target Y
# (It appears your parameter/ordering expects first col ~ steer coefficient)
# -------------------------------
Phi = np.column_stack((FWsteerRad, yawRad, LatAccMS))
Y = r_dot_meas

# -------------------------------
# SVD diagnostics, naive, pinv, truncated SVD
# -------------------------------
U, S, Vt = np.linalg.svd(Phi, full_matrices=False)
print("Singular values of Phi:\n", S)
cond = S[0] / S[-1] if S[-1] != 0 else np.inf
print("Condition number (sigma1 / sigma_n):", cond)

# naive inversion (can blow up)
theta_svd_naive = Vt.T @ np.diag(1.0 / S) @ U.T @ Y
print("\nTheta (naive SVD inversion):", theta_svd_naive)
A1 = theta_svd_naive[0]
print("A1 (naive) =", A1)
if abs(A1) < 1e-12:
    print("A1 is essentially zero -> unstable for Iz estimation.")

# pseudoinverse
theta_pinv = np.linalg.pinv(Phi) @ Y
print("\nTheta (np.linalg.pinv):", theta_pinv)
A1_pinv = theta_pinv[0]
Iz_est_pinv = a * Cf / A1_pinv if abs(A1_pinv) > 1e-12 else np.nan
print("Estimated I_z (pinv) = ", Iz_est_pinv)

# truncated SVD (regularization)
tol = 1e-6 * S[0]
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
# Plot measured vs predicted yaw acceleration (pinv)
# -------------------------------
r_dot_pred_pinv = Phi @ theta_pinv

plt.figure(figsize=(10, 4))
plt.plot(t, Y, label='Measured yaw acceleration', alpha=0.6)
plt.plot(t, r_dot_pred_pinv, '--', label='Model prediction (pinv fit)', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Yaw acceleration [rad/s²]')
plt.legend()
plt.title('Yaw Acceleration Fit (pinv)')
plt.grid(True)
plt.tight_layout()
plt.show()
