# simulation_main.py
import numpy as np
from true_path import generate_true_path
from add_noise import generate_gps, generate_imu, generate_optical_flow
from ekf import SimpleEKF
from visualization import visualize_simulation

# reproducible
np.random.seed(2)

def main():
    total_time = 25.0
    dt = 0.1
    t, x_true, y_true, vx_true, vy_true = generate_true_path(total_time, dt)

    gps_x, gps_y = generate_gps(x_true, y_true, gps_sigma=0.8, jump_prob=0.02)
    ax_imu, ay_imu = generate_imu(vx_true, vy_true, dt=dt, imu_acc_sigma=0.15)
    of_x, of_y = generate_optical_flow(x_true, y_true, dt=dt, flow_sigma=0.04, drop_prob=0.12)

    ekf = SimpleEKF(dt)
    # set a poor initial position (shows correction)
    ekf.x[:2] = np.array([x_true[0] + 1.0, y_true[0] - 1.0])
    ekf.x[2:] = np.array([0.0, 0.0])

    # call visualization (runs animation and telemetry)
    visualize_simulation(t, x_true, y_true, gps_x, gps_y, ax_imu, ay_imu, of_x, of_y, ekf)

if __name__ == "__main__":
    main()
