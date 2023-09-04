# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
from utils.rboxs_utils import rbox2poly
from mmcv.ops import box_iou_rotated
import torch

def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)




def iou_eight(bbox, candidates):

    a = np.array(bbox).reshape(4, 2)  # 四边形二维坐标表示,四行两列
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    #print('凸包值',Polygon(a).convex_hull)  # 可以打印看看是不是这样子
    b = np.array(candidates).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
 
    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = MultiPoint(union_poly).convex_hull.area

            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        col=0
        for i in detection_indices:
            # bbox = tracks[track_idx].mean[:5]
            # candidates = np.asarray(detections[i].rbox)
            # 往代价矩阵里依次添加交并比阈值，iou_eight为上面定义的函数，跟踪框和检测框的交并比
            # cost_matrix[row, col] = 1. - iou_eight(rbox2poly(bbox), rbox2poly(candidates)) # Bbox为跟踪框，candidates为检测框

            candidates = torch.tensor(detections[i].rbox).unsqueeze(0)
            bbox = torch.tensor(tracks[track_idx].mean[:5],device=candidates.device,dtype=candidates.dtype).unsqueeze(0)
            
            cost_matrix[row, col] = 1. - box_iou_rotated( bbox, candidates)
            col=col+1
    return cost_matrix
