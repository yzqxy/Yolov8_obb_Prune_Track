import numpy as np
import torch
from .sort.tracker import Tracker

__all__ = ['Byte_tracker']


class Byte_tracker(object):
    def __init__(self, 
                 max_iou_distance=0.5,
                 max_age=70,
                 n_init=3,
                ):
        
        self.tracker = Tracker( max_iou_distance=max_iou_distance, 
                        max_age=max_age, n_init=n_init)

    def update(self, rbox, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # 根据检测框的分数将框分为高质量框det和低质量框det_second,阈值可自行设定
        track_high_thresh=0.5
        track_low_thresh=0.1
        det=[]
        det_second=[]

        for i, conf in enumerate(confidences):
            if conf >= track_high_thresh:
                det.append([rbox[i],confidences[i],classes[i]])
            elif conf > track_low_thresh :
                det_second.append([rbox[i],confidences[i],classes[i]])

        # update tracker
        self.tracker.predict()
        self.tracker.update(det, det_second)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            #该track处于未证实状态或者是time_since_update大于3帧则跳出本次循环
            if not track.is_confirmed() or track.time_since_update > 3:
                continue
            rbox = track.mean[:5]
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append([rbox, track_id, class_id, np.array(conf.cpu())])
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs


