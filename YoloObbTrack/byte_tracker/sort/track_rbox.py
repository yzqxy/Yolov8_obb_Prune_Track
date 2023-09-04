# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
from .kalman_filter_rbox import KalmanFilter_Rbox


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, rbox, track_id, conf, class_id, n_init, max_age):
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.rbox = rbox
        self.state = TrackState.Tentative

        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter_Rbox()
        self.mean, self.covariance = self.kf.initiate(rbox)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        # ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        # ret[:2] -= ret[2:] / 2

        ret = self.mean[:8].copy()
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret



    # def increment_age(self):
    #     self.age += 1
    #     self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        # self.age += 1
        self.time_since_update += 1

    def update(self, rbox,  conf,class_id):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.conf = conf
        self.class_id = class_id
        self.rbox = rbox
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, rbox, conf)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative and self.time_since_update > 3:
            self.state = TrackState.Deleted
        if self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
