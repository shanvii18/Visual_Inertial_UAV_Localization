# add_noise.py
import numpy as np

def generate_gps(x_true, y_true, gps_sigma=0.8, jump_prob=0.02):
    """
    GPS position with gaussian noise and occasional large spikes (outliers).
    """
    N = len(x_true)
    gps_x = x_true + np.random.normal(0, gps_sigma, size=N)
    gps_y = y_true + np.random.normal(0, gps_sigma, size=N)
    spikes = (np.random.rand(N) < jump_prob)
    if spikes.any():
        gps_x[spikes] += np.random.normal(0, 6.0, size=spikes.sum())
        gps_y[spikes] += np.random.normal(0, 6.0, size=spikes.sum())
    return gps_x, gps_y

def generate_imu(vx, vy, dt=0.1, imu_acc_sigma=0.15):
    """
    IMU gives approximate accelerations (derivative of velocity) + noise
    """
    ax = np.gradient(vx, dt) + np.random.normal(0, imu_acc_sigma, size=len(vx))
    ay = np.gradient(vy, dt) + np.random.normal(0, imu_acc_sigma, size=len(vy))
    return ax, ay

def generate_optical_flow(x_true, y_true, dt=0.1, flow_sigma=0.04, drop_prob=0.08):
    """
    Optical flow returns small frame-to-frame displacement (Δx, Δy) with noise and random dropouts.
    """
    dx = np.diff(x_true, prepend=x_true[0])
    dy = np.diff(y_true, prepend=y_true[0])
    of_x = dx + np.random.normal(0, flow_sigma, size=len(dx))
    of_y = dy + np.random.normal(0, flow_sigma, size=len(dy))
    drops = (np.random.rand(len(dx)) < drop_prob)
    of_x[drops] = np.nan
    of_y[drops] = np.nan
    return of_x, of_y
