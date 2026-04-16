# visualization.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np
import time
import matplotlib.animation as animation

def visualize_simulation(t, x_true, y_true, gps_x, gps_y, ax_imu, ay_imu, of_x, of_y, ekf):
    """
    Creates animation window with:
    - True path (light blue)
    - GPS noisy dots (grey)
    - EKF filtered path (red)
    - Drone icon
    - Optical flow arrows near drone
    - Telemetry panel at right
    ekf: instance of SimpleEKF (pre-initialized)
    """
    N = len(t)
    dt = ekf.dt

    fig = plt.figure(figsize=(11,6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2,2,1], wspace=0.25)
    ax_map = fig.add_subplot(gs[:, :2])
    ax_panel = fig.add_subplot(gs[:, 2])

    ax_map.set_title("Visual–Inertial UAV Localization (Option B: Grid background)")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.grid(True, linestyle=':', color='0.85')

    margin = 4.0
    xmin, xmax = x_true.min()-margin, x_true.max()+margin
    ymin, ymax = y_true.min()-margin, y_true.max()+margin
    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)

    ax_map.plot(x_true, y_true, lw=1.0, color='lightsteelblue', label='True path (reference)')

    gps_dots, = ax_map.plot([], [], linestyle='None', marker='o', markersize=4, color='grey', alpha=0.6, label='GPS noisy')
    ekf_line, = ax_map.plot([], [], lw=2.2, color='red', label='EKF filtered')
    drone_artist = Circle((0,0), radius=0.25, color='black', fill=True, zorder=5)
    ax_map.add_patch(drone_artist)
    of_arrows = []

    ax_panel.axis('off')
    txt_time = ax_panel.text(0.01, 0.90, "", transform=ax_panel.transAxes, fontsize=10)
    txt_gps = ax_panel.text(0.01, 0.78, "", transform=ax_panel.transAxes, fontsize=10)
    txt_imu = ax_panel.text(0.01, 0.66, "", transform=ax_panel.transAxes, fontsize=10)
    txt_flow = ax_panel.text(0.01, 0.54, "", transform=ax_panel.transAxes, fontsize=10)
    txt_ekf = ax_panel.text(0.01, 0.42, "", transform=ax_panel.transAxes, fontsize=10)

    ekf_hist_x = []
    ekf_hist_y = []
    gps_hist_x = []
    gps_hist_y = []

    def init():
        gps_dots.set_data([], [])
        ekf_line.set_data([], [])
        drone_artist.center = (x_true[0], y_true[0])
        return gps_dots, ekf_line, drone_artist

    def update(i):
        nonlocal of_arrows
        ax_i = ax_imu[i]
        ay_i = ay_imu[i]
        gps_i = (gps_x[i], gps_y[i])
        of_i = (of_x[i], of_y[i])

        ekf.predict(ax_i, ay_i)

        innov = np.linalg.norm(np.array(gps_i) - ekf.x[:2])
        if not np.isnan(gps_i).any():
            if innov < 8.0:
                ekf.update_gps(gps_i)

        if not (np.isnan(of_i[0]) or np.isnan(of_i[1])):
            ekf.update_flow(of_i)

        ekf_hist_x.append(ekf.x[0])
        ekf_hist_y.append(ekf.x[1])
        gps_hist_x.append(gps_i[0])
        gps_hist_y.append(gps_i[1])

        gps_dots.set_data(gps_hist_x, gps_hist_y)
        ekf_line.set_data(ekf_hist_x, ekf_hist_y)
        drone_artist.center = (ekf.x[0], ekf.x[1])

        for a in of_arrows:
            a.remove()
        of_arrows = []

        if not (np.isnan(of_i[0]) or np.isnan(of_i[1])):
            arrow = FancyArrowPatch(
                (ekf.x[0], ekf.x[1]),
                (ekf.x[0] + of_i[0]*8.0, ekf.x[1] + of_i[1]*8.0),
                arrowstyle='->', mutation_scale=8, linewidth=1.2
            )
            ax_map.add_patch(arrow)
            of_arrows.append(arrow)

        elapsed = i * dt
        txt_time.set_text(f"Time: {elapsed:.2f} s")
        txt_gps.set_text(f"GPS Position: ({gps_i[0]:.2f}, {gps_i[1]:.2f})")
        txt_imu.set_text(f"IMU accel: ax={ax_i:.2f} m/s², ay={ay_i:.2f} m/s²")
        if not (np.isnan(of_i[0]) or np.isnan(of_i[1])):
            txt_flow.set_text(f"Vision (Optical Flow Δ): ({of_i[0]:.3f} , {of_i[1]:.3f})")
        else:
            txt_flow.set_text("Vision (Optical Flow Δ): (no meas)")
        txt_ekf.set_text(f"EKF Output (x,y,vx,vy): ({ekf.x[0]:.2f}, {ekf.x[1]:.2f}, {ekf.x[2]:.2f}, {ekf.x[3]:.2f})")

        return gps_dots, ekf_line, drone_artist, txt_time, txt_gps, txt_imu, txt_flow, txt_ekf

    anim = animation.FuncAnimation(fig, update, frames=N, init_func=init,
                                   interval=dt*1000, blit=False, repeat=False)

    ax_map.legend(loc='upper left')
    plt.tight_layout()
    # To save video: caller can uncomment and supply writer (ffmpeg)
    # writer = animation.FFMpegWriter(fps=int(1/dt))
    # anim.save('simulation_output.mp4', writer=writer)
    plt.show()
