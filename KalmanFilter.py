import numpy as np
import math
import time
import helpers


qv = 6.15  # std_speed/2 : variance on speed state
qw = 6.15  # std_speed/2 : angle speed also depends
rv = 6.15  # std_speed/2
rw = 6.15  # std_speed/2
# the positions variances
qpx = 0.04
qpy = 0.04
qth = 0.01
rpx = 0.1
rpy = 0.1
rth = 0.1
Q = np.array(
    [
        [qpx, 0, 0, 0, 0],
        [0, qpy, 0, 0, 0],
        [0, 0, qth, 0, 0],
        [0, 0, 0, qv, 0],
        [0, 0, 0, 0, qw],
    ]
)
# matrice H and R with camera presence
H_1 = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)
R_1 = np.array(
    [
        [rpx, 0, 0, 0, 0],
        [0, rpy, 0, 0, 0],
        [0, 0, rth, 0, 0],
        [0, 0, 0, rv, 0],
        [0, 0, 0, 0, rw],
    ]
)
# matrice H and R with no camera presence
H_2 = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
R_2 = np.array([[rv, 0], [0, rw]])


def kalman_filter(Ts, speed, theta_speed, curr_pos, x_est_prev, P_est_prev):

    """
    Estimates the current state using input sensor data and the previous state

    param speed: measured speed (Thymio units)
    param theta_speed: the rotation speed
    param curr_pos: measured the position (x, y, theta)
    param x_est_prev: previous state a posteriori estimation
    param P_est_prev: previous state a posteriori covariance

    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """
    ## Prediciton through the a priori estimate
    # estimated mean of the state
    # calc matrix A
    phy = math.radians(curr_pos[0]) if curr_pos[0] is not None else x_est_prev[2][0]
    cos_phy = np.cos(phy)
    sin_phy = np.sin(phy)
    A = np.array(
        [
            [1, 0, 0, 2 * Ts * cos_phy, 0],
            [0, 1, 0, 2 * Ts * sin_phy, 0],
            [0, 0, 1, 0, -2 * Ts],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    x_est_a_priori = np.dot(A, x_est_prev)
    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori

    ## Update
    # y, C, and R for a posteriori estimate, depending on the presence of the camera

    if curr_pos[0] is None:
        y = np.array([[speed], [theta_speed]])
        H = H_2
        R = R_2

    else:
        # x,y,theta,speed,theta speed
        y = np.array(
            [
                [curr_pos[1][0]],
                [curr_pos[1][1]],
                [math.radians(curr_pos[0])],
                [speed],
                [theta_speed],
            ]
        )
        H = H_1
        R = R_1

    # innovation / measurement residual
    i = y - np.dot(H, x_est_a_priori)
    # measurement prediction covariance
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)))

    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K, i)
    P_est = P_est_a_priori - np.dot(K, np.dot(H, P_est_a_priori))

    return x_est, P_est


class KalmanFilterManager:
    def __init__(self, phy_0, initial_coords, node):
        self.node = node
        self.Ts = 0.1
        # variable
        self.Ts = 0.01
        self.d_wheel = 33  # between wheels in mm
        self.speed_conv_factor = 0.3375 / 3
        self.last = time.time()

        # calc matrix A
        phy_0_rads = math.radians(phy_0)
        self.A = None

        # calculate Q and R state and measurement errors for Kalman:

        # the speeds variances

        # state vector [x, y, theta, speed, theta_speed]
        self.x_est = [
            np.array([[initial_coords[0]], [initial_coords[1]], [phy_0_rads], [0], [0]])
        ]
        self.P_est = [0 * np.ones(5)]
        self.robot_pos = [int(self.x_est[0][0][0]), int(self.x_est[0][1][0])]
        self.robot_angle = math.degrees(self.x_est[0][2][0])

    def update_position(self, curr_pos):
        wheel_r = self.node["motor.right.speed"] * self.speed_conv_factor
        wheel_l = self.node["motor.left.speed"] * self.speed_conv_factor

        speed = (wheel_r + wheel_l) / 2
        theta_speed = (wheel_r - wheel_l) / (2 * self.d_wheel)
        # update the delta time betwwen last calculations
        now = time.time()
        Ts = now - self.last
        self.last = now
        # get new state estimation
        x_est_new, p_est_new = kalman_filter(
            Ts, speed, theta_speed, curr_pos, self.x_est[-1], self.P_est[-1]
        )
        x_est_new[2][0] = math.radians(
            helpers.angle_modulo(math.degrees(x_est_new[2][0]))[0]
        )
        self.x_est.append(x_est_new)
        self.P_est.append(p_est_new)
        self.robot_pos = [int(x_est_new[0][0]), int(x_est_new[1][0])]
        self.robot_angle = math.degrees(x_est_new[2][0])
        return self.robot_pos, self.robot_angle
