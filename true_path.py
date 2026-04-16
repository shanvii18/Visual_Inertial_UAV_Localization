# true_path.py
import numpy as np

def generate_true_path(total_time=25.0, dt=0.1):
    """
    Returns: t, x, y, vx, vy
    Same smooth path as used in the single-file demo.
    """
    t = np.arange(0, total_time + 1e-9, dt)
    x = 0.5 * t + 2.0 * np.sin(0.5 * t)
    y = 0.3 * t + 1.5 * np.cos(0.3 * t)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    return t, x, y, vx, vy
