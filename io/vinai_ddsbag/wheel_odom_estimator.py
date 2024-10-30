import numpy as np


class WheelOdomEstimator:
    def __init__(self, config=None):
        # Initialize state and matrices
        self.mu = np.zeros(6)
        self.zt = np.zeros(4)
        self.I_mat = np.eye(6)
        self.Sigma_sq = np.zeros((6, 6))
        self.Relative_Sigma_sq = np.zeros((6, 6))
        self.Init_relative_sigma_sq = np.zeros((6, 6))
        self.Sigma_msq = np.zeros((6, 6))
        self.Sigma_z = np.zeros((4, 4))
        self.H = np.zeros((4, 6))
        self.Fk = np.eye(6)

        # Parameters
        self.wheel_base = 0.0
        self.steering_angle_factor = 0.0
        self.wheel_radius = 0.0
        self.traj_logged = False

        # Timing and state variables
        self.time_now = 0.0
        self.time_last = 0.0
        self.dt = 0.0

        if config is not None:
            assert len(config['sigma_sq']) == 6
            assert len(config['sigma_msq']) == 6
            for i in range(6):
                self.Sigma_sq[i, i] = config['sigma_sq'][i]
                self.Sigma_msq[i, i] = config['sigma_msq'][i]

            assert len(config['sigma_z']) == 4
            for i in range(4):
                self.Sigma_z[i, i] = config['sigma_z'][i]

            self.wheel_base = config['wheel_base']
            self.wheel_radius = config['wheel_radius']
            self.steering_angle_factor = config['steering_angle_factor']

        self.init()

    def update_info_and_estimate(self, speed_timestamp, gear, wheel_speed, yaw_rate, steering_angle):
        self.predict(speed_timestamp, gear, wheel_speed, yaw_rate, steering_angle)
        self.update()

    def predict(self, speed_timestamp, gear, wheel_speed, yaw_rate, steering_angle):
        yaw_rate = 0 if abs(yaw_rate < 0.01) else yaw_rate

        self.time_now = speed_timestamp / 1e9

        if self.time_last > 0:
            self.dt = (self.time_now - self.time_last)

        self.time_last = self.time_now

        # Update odometry state with sensor data
        u_odom = (wheel_speed[2] + wheel_speed[3]) * self.wheel_radius / 2.0
        u_odom = - u_odom if gear == 2 else u_odom

        v_imu = 0.0

        omega_steering = u_odom * np.tan(steering_angle / self.steering_angle_factor * np.pi / 180.0) / self.wheel_base
        omega_imu = yaw_rate * np.pi / 180.0

        # Current state variables
        theta = self.mu[2]
        u = self.mu[3]
        v = self.mu[4]
        omega = self.mu[5]

        # Predict the next state
        predict = np.array([
            u * np.cos(theta) - v * np.sin(theta),
            u * np.sin(theta) + v * np.cos(theta),
            omega, 0, 0, 0
        ])
        self.mu += predict * self.dt

        # Update the state transition matrix Fk
        self.Fk[0, 2] = (-u * np.sin(theta) - v * np.cos(theta)) * self.dt
        self.Fk[0, 3] = np.cos(theta) * self.dt
        self.Fk[0, 4] = -np.sin(theta) * self.dt
        self.Fk[1, 2] = (u * np.cos(theta) - v * np.sin(theta)) * self.dt
        self.Fk[1, 3] = np.sin(theta) * self.dt
        self.Fk[1, 4] = np.cos(theta) * self.dt
        self.Fk[2, 5] = self.dt

        # Update observation vector zt
        self.zt = np.array([u_odom, v_imu, omega_imu, omega_steering])

    def update(self):
        # Update covariance matrices
        self.Sigma_sq = self.Fk @ self.Sigma_sq @ self.Fk.T + self.Sigma_msq
        self.Relative_Sigma_sq = self.Fk @ self.Init_relative_sigma_sq @ self.Fk.T + self.Sigma_msq

        # Compute the measurement covariance
        covariance = self.H @ self.Sigma_sq @ self.H.T + self.Sigma_z

        # Compute the inverse of the covariance matrix if it is non-singular
        if np.abs(np.linalg.det(covariance)) > 1e-6:
            covariance_inv = np.linalg.inv(covariance)
        else:
            covariance_inv = np.zeros_like(covariance)

        # Compute the Kalman gain
        Kt = self.Sigma_sq @ self.H.T @ covariance_inv

        # Update the state estimate
        measurement = self.zt - self.H @ self.mu
        self.mu += Kt @ measurement
        self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))

        # Update the covariance matrices
        self.Sigma_sq = (self.I_mat - Kt @ self.H) @ self.Sigma_sq
        self.Relative_Sigma_sq = (self.I_mat - Kt @ self.H) @ self.Relative_Sigma_sq

    def get_current_pose(self):
        # Extract the current pose (x, y, theta)
        x = self.mu[0]
        y = self.mu[1]
        theta = self.mu[2]
        return x, y, theta

    def get_pose_at_time(self, timestamp):
        dt = timestamp / 1e9 - self.time_now
        u = self.mu[3]
        v = self.mu[4]
        w = self.mu[5]
        theta_now = self.mu[2]
        x = self.mu[0] + u * np.cos(theta_now) * dt - v * np.sin(theta_now) * dt
        y = self.mu[1] + u * np.sin(theta_now) * dt + v * np.cos(theta_now) * dt
        theta = theta_now + w * dt

        return x, y, theta

    def init(self):
        self.H[0, 3] = 1.0
        self.H[1, 4] = 1.0
        self.H[2, 5] = 1.0
        self.H[3, 5] = 1.0
