# ekf.py
import numpy as np

class SimpleEKF:
    """
    Simple Extended Kalman Filter for 2D constant-velocity model.
    State vector: [x, y, vx, vy]
    Predict uses IMU accelerations.
    Update uses GPS (x,y) and optical-flow Δpos (treated as velocity*dt measurement).
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x = np.array([0., 0., 0., 0.])   # x, y, vx, vy
        self.P = np.eye(4) * 1.0
        q_pos = 0.01
        q_vel = 0.5
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R_gps = np.diag([0.8**2, 0.8**2])
        self.R_flow = np.diag([0.06**2, 0.06**2])

    def predict(self, ax, ay):
        dt = self.dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        B = np.array([[0.5*dt*dt, 0],
                      [0, 0.5*dt*dt],
                      [dt, 0],
                      [0, dt]])
        u = np.array([ax, ay])
        self.x = F.dot(self.x) + B.dot(u)
        self.P = F.dot(self.P).dot(F.T) + self.Q

    def update_gps(self, z_pos):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        z = np.array(z_pos)
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + self.R_gps
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(H)).dot(self.P)

    def update_flow(self, z_dpos):
        """
        z_dpos : measured delta position (Δx, Δy) between frames.
        Approximate measurement relation: Δpos ≈ [vx*dt, vy*dt], so H maps state->Δpos.
        """
        dt = self.dt
        H = np.array([[0, 0, dt, 0],
                      [0, 0, 0, dt]])
        z = np.array(z_dpos)
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + self.R_flow
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(H)).dot(self.P)
