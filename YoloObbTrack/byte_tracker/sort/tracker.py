# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter_rbox
from . import linear_assignment
from . import iou_matching
from .track_rbox import Track
# from .track import Track

class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    GATING_THRESHOLD = np.sqrt(kalman_filter_rbox.chi2inv95[4])

    def __init__(self,  max_iou_distance=0.7, max_age=30, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter_rbox.KalmanFilter_Rbox()
        self.tracks = []
        self._next_id = 1

    def _initiate_track(self, rbox,conf, class_id):
        self.tracks.append(Track(rbox, self._next_id,conf, class_id, self.n_init, self.max_age))
        self._next_id += 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)


    def update(self, det, det_scond):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        #get confirmed_tracks and  unconfirmed_tracks indices
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        #step1,计算高质量检测框与已经激活的跟踪框的匹配结果
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost_fuse_score, self.max_iou_distance, self.tracks,
                det, confirmed_tracks)
        #是否要融合得分值看数据需求，测试版本并未使用iou_matching.iou_cost_fuse_score
        # matches_a, unmatched_tracks_a, unmatched_detections_a = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         det, confirmed_tracks)
   
        # 更新step1中与高质量检测框匹配的跟踪框状态
        for track_idx, detection_idx in matches_a:
            self.tracks[track_idx].update(
                det[detection_idx][0], det[detection_idx][1], det[detection_idx][2])

        # step2,计算低质量框的检测结果与第一步未匹配到的跟踪框的匹配结果
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, 0.5, self.tracks , det_scond, unmatched_tracks_a)
     
        # 更新step2已匹配框的状态
        for track_idx, detection_idx in matches_b:
            self.tracks[track_idx].update(
                det_scond[detection_idx][0], det_scond[detection_idx][1], det_scond[detection_idx][2])
        # step2未匹配到的框状态置为missed
        for track_idx in unmatched_tracks_b:
            self.tracks[track_idx].mark_missed()

        #step3 将step1中未匹配上的检测框和unconfirmed_tracks中的框进行匹配
        matches_c, unmatched_tracks_c, unmatched_detections_c = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, 0.7,  self.tracks,det ,unconfirmed_tracks,unmatched_detections_a)

        # 更新step3已匹配框的状态
        for track_idx, detection_idx in matches_c:
            self.tracks[track_idx].update(
                det[detection_idx][0], det[detection_idx][1], det[detection_idx][2])
        # step3未匹配到的框状态置为missed
        for track_idx in unmatched_tracks_c:
            self.tracks[track_idx].mark_missed()
        #从上述3个step过程中都未匹配到的检测框，我们将它认定为一个新的轨迹进行初始化。
        for detection_idx in unmatched_detections_c:
            self._initiate_track(det[detection_idx][0], det[detection_idx][1], det[detection_idx][2])
        #从跟踪队列里剔除删除状态的跟踪框
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
  

