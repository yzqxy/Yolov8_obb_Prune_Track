# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter_Rbox(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 16-dimensional state space

        x1, y1, x2, y2,x3, y3,x4, y4, vx1, vy1, vx2, vy2,vx3, vy3,vx4, vy4

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    a=np.eye(4,k=1)
    print(a)
    
    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]
    """

    def __init__(self):
        ndim, dt = 5, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty inW
        # the model. This is a bit hacky.WW
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        measurement=measurement
        mean_pos = measurement  #检测目标的位置[旋转框， x, y, w, h, theta]
        mean_vel = np.zeros_like(mean_pos) #刚出现的新目标默认其速度=0,构造一个与box维度一样的向量 [0. 0. 0. 0. 0.]
        mean = np.r_[mean_pos, mean_vel]  # 是按列连接两个矩阵 [ x, y, w, h, theta, 0. 0. 0. 0. 0.]
        # print('mean_init',mean)

        #协方差矩阵，元素值越大，表明不确定性越大，可以以任意值初始化
        std = [
            2 * self._std_weight_position * measurement[0],   # the center point x
            2 * self._std_weight_position * measurement[1],   # the center point y
            1 * (measurement[2]/measurement[3]),                               # the ratio of width/height
            2 * self._std_weight_position * measurement[3],   # the height
            2 * self._std_weight_position * measurement[4],   # the theta
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * (measurement[2]/measurement[3]),
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[4]]
        
        covariance = np.diag(np.square(std))  #np.square(std)对std中的每个元素平方，np.diag构成一个10×10的对角矩阵，对角线上的元素是np.square(std)
        return mean, covariance
    
    # 相当于得到t时刻估计值
    def predict(self, mean, covariance):
        # Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 *(mean[2]/ mean[3]),
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[4]]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * (mean[2]/ mean[3]),
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[4]]
        # np.r_ 按列连接两个矩阵
        # 初始化噪声矩阵Q=motion_cov
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        # 预测当前帧框的位置x' = Fx
        mean = np.dot(self._motion_mat, mean)
        # P' = FPF^T+Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance,confidence=.0):
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
            1e-3]

        std = [(1 - confidence) * x for x in std]
        # 初始化噪声矩阵R
        innovation_cov = np.diag(np.square(std))
        # 将均值向量映射到检测空间，即Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov
    
    # 通过估计值和观测值估计最新结果
    def update(self, mean, covariance, measurement, confidence=.0):
        measurement=measurement
        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        projected_mean, projected_cov = self.project(mean, covariance,confidence)
        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx' 当前帧检测框减去卡尔曼预测的框位置获得的偏移量
        innovation = measurement - projected_mean  #
        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance



    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
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
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        # print('mean',mean)
        # print('measurements',measurements)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
