"""
Implementation of Kalman filter specifically designed for tracking objects
using position (x, y), scale (s) and aspect ratio (r).
""" 

from typing import Tuple
from copy import deepcopy
from collections import deque
from math import log, exp
from numpy import (
    dot, zeros, eye, isscalar, array,
    log as np_log, exp as np_exp, sqrt
)
import numpy as np
from numpy.linalg import inv
from boxmot.motion.kalman_filters.aabb.base_kalman_filter import BaseKalmanFilter

def reshape_z(z, dim_z):
    """Ensure z is a (dim_z, 1) shaped vector."""
    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T
    if z.shape != (dim_z, 1):
        raise ValueError(f'z must be convertible to shape ({dim_z}, 1)')
    return z

def logpdf(x, mean, cov):
    """Compute log of multivariate normal distribution."""
    dim = x.shape[0]
    diff = x - mean
    mahal = dot(dot(diff.T, inv(cov)), diff)[0, 0]
    return -0.5 * (dim * np_log(2 * np.pi) + np_log(np.linalg.det(cov)) + mahal)



class KalmanFilterXYSR(BaseKalmanFilter):
    """
    A Kalman filter for tracking bounding boxes in image space.
    The state space is `(x, y, s, r, vx, vy, vs, vr)` containing
    the center position (x, y), scale s, ratio r and their respective velocities.
    """

    def __init__(self):
        super().__init__(ndim=8)  # State space: x, y, s, r, vx, vy, vs, vr
        
        # Define constant velocity model matrices (8x8 matrix for state [x, y, s, r, vx, vy, vs, vr])
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0, 1],  # r = r + vr
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vs = vs
            [0, 0, 0, 0, 0, 0, 0, 1],  # vr = vr
        ])

        # Measurement matrix - 4x8 matrix since we only observe [x, y, s, r]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # observe x
            [0, 1, 0, 0, 0, 0, 0, 0],  # observe y
            [0, 0, 1, 0, 0, 0, 0, 0],  # observe s
            [0, 0, 0, 1, 0, 0, 0, 0],  # observe r
        ])
        
        self.dim_z = self.H.shape[0]

        # Initialize other required matrices
        self.B = None  # No control input
        self.Q = None  # Process noise (computed in predict)
        self.R = None  # Measurement noise (computed in update)
        self._I = np.eye(8)  # Identity matrix
        
        # Initialize state tracking
        self.max_obs = 50  # Maximum number of observations to keep
        self.history_obs = deque([], maxlen=self.max_obs)
        self.observed = False
        self.last_measurement = None
        self.attr_saved = None
        
    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, s, r) with center position (x, y),
            scale s, and aspect ratio r.
        """
        # Initial state (8 dimensional)
        mean_pos = measurement.flatten()
        mean_vel = np.zeros_like(mean_pos)
        self.x = np.r_[mean_pos, mean_vel].reshape((8, 1))

        # Initial uncertainty based on measurement
        std = self._get_initial_covariance_std(measurement)
        self.P = np.diag(np.square(std))

        # Initialize history
        self.history_obs = deque([], maxlen=50)
        self.history_obs.append(measurement.flatten())

    def _get_initial_covariance_std(self, measurement: np.ndarray) -> np.ndarray:
        meas = measurement.flatten()
        # Initialize with reasonable standard deviations
        return np.array([
            2 * self._std_weight_position * meas[2],  # x
            2 * self._std_weight_position * meas[2],  # y
            2 * self._std_weight_position * meas[2],  # s (scale)
            1e-2,                                     # r (ratio)
            10 * self._std_weight_velocity * meas[2], # vx
            10 * self._std_weight_velocity * meas[2], # vy
            10 * self._std_weight_velocity * meas[2], # vs (scale velocity)
            1e-5                                      # vr (ratio velocity)
        ])

    def _get_process_noise_std(self, mean: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_flat = mean.flatten()
        # Process noise for position/scale and velocity
        std_pos = np.array([
            self._std_weight_position * mean_flat[2],  # x
            self._std_weight_position * mean_flat[2],  # y
            self._std_weight_position * mean_flat[2],  # s
            1e-2                                       # r
        ])
        std_vel = np.array([
            self._std_weight_velocity * mean_flat[2],  # vx
            self._std_weight_velocity * mean_flat[2],  # vy
            self._std_weight_velocity * mean_flat[2],  # vs
            1e-5                                       # vr
        ])
        return std_pos, std_vel

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # Project state using measurement matrix
        projected_mean = dot(self.H, mean)
        projected_cov = dot(dot(self.H, covariance), self.H.T)
        return projected_mean, projected_cov

    def gating_distance(self, mean, covariance, measurements,
                       only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, s, r) where (x, y) is the center position, s is the
            scale (area), and r is the aspect ratio.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the (x, y)
            position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean, projected_cov = projected_mean[:2], projected_cov[:2, :2]
            measurements = measurements[:, :2]

        # Ensure projected_mean is a column vector
        projected_mean = np.atleast_2d(projected_mean).reshape(-1, 1)

        # Calculate Mahalanobis distances for all measurements
        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean.T
        z = np.linalg.solve(cholesky_factor, d.T)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha

    def _get_measurement_noise_std(self, mean: np.ndarray, confidence: float) -> np.ndarray:
        mean_flat = mean.flatten()
        # Measurement noise adjusted by detection confidence
        std = np.array([
            self._std_weight_position * mean_flat[2],  # x
            self._std_weight_position * mean_flat[2],  # y
            self._std_weight_position * mean_flat[2],  # s
            1e-1                                       # r
        ])
        
        # Apply confidence scaling (lower confidence = higher noise)
        if confidence > 0:
            std = std * (2 - confidence)
        return std

    def apply_affine_correction(self, m, t):
        """
        Apply to both last state and last observation for OOS smoothing.

        Messy due to internal logic for kalman filter being messy.
        """
        t_col = t.reshape(-1, 1)
        self.x[:2] = m @ self.x[:2] + t_col
        self.x[4:6] = m @ self.x[4:6]

        self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
        self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T

        # If frozen, also need to update the frozen state for OOS
        if not self.observed and self.attr_saved is not None:
            self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t_col
            self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]

            self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
            self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T

            lm_pos = self.attr_saved["last_measurement"][:2].reshape(2, 1)
            self.attr_saved["last_measurement"][:2] = (m @ lm_pos + t_col).flatten()

    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations. This implementation specializes the prediction for XYSR tracking
        by computing an appropriate process noise matrix Q based on the current state.
        """
        # Get process noise based on current state
        std_pos, std_vel = self._get_process_noise_std(self.x)
        
        # Create process noise matrix
        Q = np.diag(np.concatenate([np.square(std_pos), np.square(std_vel)]))
        
        # Predict state: x = Fx
        self.x = dot(self.F, self.x)
        
        # Predict covariance: P = FPF' + Q
        self.P = dot(dot(self.F, self.P), self.F.T) + Q
        
        # Save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def freeze(self):
        """
        Save the parameters before non-observation forward
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(list(self.history_obs))
            self.__dict__ = self.attr_saved
            self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1, index2 = indices[-2], indices[-1]
            box1, box2 = new_history[index1], new_history[index2]
            x1, y1, s1, r1 = box1
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

            for i in range(index2 - index1):
                x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
                w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
                s, r = w * h, w / float(h)
                new_box = np.array([x, y, s, r])
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(self, measurement: np.ndarray, confidence: float = 1.0):
        """
        Update the state estimate with a new measurement.
        
        Parameters
        ----------
        measurement : ndarray
            The measurement vector (x, y, s, r)
        confidence : float, optional
            Detection confidence in [0,1]. Higher values -> lower noise.
            Default is 1.0 (perfect detection).
        """
        if measurement is None:
            if self.observed:
                # No observation, freeze current state for potential smoothing
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.history_obs.append(None)
            return

        # If coming back from unobserved state, apply smoothing
        if not self.observed:
            self.unfreeze()
        self.observed = True

        # Reshape measurement to column vector if needed
        z = reshape_z(measurement, self.dim_z)
        
        # Compute measurement noise based on current state and confidence
        std = self._get_measurement_noise_std(self.x, confidence)
        R = np.diag(np.square(std))

        # Pre-fit residual
        y = z - dot(self.H, self.x)

        # Common subexpressions
        PHT = dot(self.P, self.H.T)
        S = dot(self.H, PHT) + R
        
        # Kalman gain
        K = dot(PHT, inv(S))

        # State update
        self.x = self.x + dot(K, y)
        
        # Covariance update using Joseph form for numerical stability
        I_KH = self._I - dot(K, self.H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(K, R), K.T)

        # Save posterior and measurement
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.history_obs.append(measurement.flatten())

    def compute_log_likelihood(self, measurement: np.ndarray) -> float:
        """
        Compute the log-likelihood of a measurement given the current state.
        
        Parameters
        ----------
        measurement : ndarray
            The measurement vector (x, y, s, r)
            
        Returns
        -------
        float
            The log-likelihood of the measurement
        """
        if measurement is None:
            return float('-inf')
        
        # Reshape measurement
        z = reshape_z(measurement, self.dim_z)
        
        # Predict measurement from current state
        predicted_measurement = dot(self.H, self.x)
        
        # Compute innovation covariance
        std = self._get_measurement_noise_std(self.x, 1.0)
        R = np.diag(np.square(std))
        S = dot(dot(self.H, self.P), self.H.T) + R
        
        return logpdf(z, predicted_measurement, S)

    def compute_likelihood(self, measurement: np.ndarray) -> float:
        """
        Compute the likelihood of a measurement given the current state.
        
        Parameters
        ----------
        measurement : ndarray
            The measurement vector (x, y, s, r)
            
        Returns
        -------
        float
            The likelihood of the measurement
        """
        return exp(self.compute_log_likelihood(measurement))


    def batch_filter(self, zs: list, Rs: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch process a sequence of measurements. This method is suitable
        for cases where the measurement noise varies with each measurement.

        Parameters
        ----------
        zs : list
            List of measurements at each time step. Missing measurements must be
            represented by None.
        Rs : list, optional
            List of measurement noise matrices. If None, uses self.R for all updates.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Returns (means, covariances) where:
            means: array of state estimates for each time step
            covariances: array of state covariances for each time step
        """
        means, covariances = [], []
        # Note: Rs is ignored in this implementation as R is computed dynamically
        # To use fixed Rs, modify update to accept R parameter
        
        for z in zs:
            self.predict()
            self.update(z)
            means.append(self.x.copy())
            covariances.append(self.P.copy())
            
        return np.array(means), np.array(covariances)