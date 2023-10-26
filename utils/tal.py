# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import LOGGER, check_version
import numpy as np
from utils.rboxs_utils import rbox2poly2
from mmcv.ops import box_iou_rotated

def check_points_in_rotated_boxes(points, boxes):
    """Check whether point is in rotated boxes

    Args:
        points (tensor): (1, L, 2) anchor points
        boxes (tensor): [B, N, 5] gt_bboxes
        eps (float): default 1e-9
    
    Returns:
        is_in_box (tensor): (B, N, L)

    """
    # [B, N, 5] -> [B, N, 4, 2]
    corners = rbox2poly2(boxes)
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = corners.split((1,1,1,1), 2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, L]
    norm_ab = torch.sum(ab * ab, dim=-1)
    # [B, N, L]
    norm_ad = torch.sum(ad * ad, dim=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, dim=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, dim=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta) 
    is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (
        ap_dot_ad <= norm_ad)
    return is_in_box



def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        # import pdb
        # pdb.set_trace()
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)  #生成one_hot形式的张量，从[8,8400]-->[8,7,8400]，7是gt数量，赋予最大iou的下标为1，其他为0

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


def rotated_iou_similarity(box1, box2):
    """Calculate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [N, 5]
        box2 (Tensor): box with the shape [N, 5]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N]
    """
    # rotated_ious = []
    # for b1, b2 in zip(box1, box2):
    #     b1=b1.unsqueeze(0)
    #     b2=b2.unsqueeze(0)
    #     rotated_ious.append(box_iou_rotated(b1, b2).squeeze(0).squeeze(0))
    # return torch.stack(rotated_ious, axis=0)
    return box_iou_rotated(box1, box2, aligned=True)

class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=1.0, eps=1e-9):
        super().__init__()
        self.topk = 13
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))


        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        # pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        norm_align_metric = (align_metric  / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        # fg_mask.bool())布尔值 True和False来判断正负样本
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        mask_in_gts = check_points_in_rotated_boxes(anc_points, gt_bboxes)

        # get anchor_align metric, (b, max_num_obj, h*w) [16, 2, 8400],[16, 2, 8400]
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,mask_in_gts * mask_gt)

        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # merge all mask to a final mask, (b, max_num_obj, h*w),mask_gt=torch.Size([16, 2, 1])
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        #是否采用欧式距离约束正负样本选择，0 false 1 true
        distance_constraint=0
        if distance_constraint==1:
            #计算每个anchor中心点与gt中心点之间的欧氏距离
            Euclidean_distance = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
            pdist = nn.PairwiseDistance(p=2)
            Euclidean_distance[mask_gt] = pdist(gt_boxes[:,:2],pd_boxes[:,:2])

            #归一化欧氏距离
            eps=0.0001
            min_score=Euclidean_distance[mask_gt].amin(0)
            max_score=Euclidean_distance[mask_gt].amax(0)
            Euclidean_distance[mask_gt]=(Euclidean_distance[mask_gt]-min_score+eps)/(max_score-min_score)
            Euclidean_distance[mask_gt]=Euclidean_distance[mask_gt].pow(0.1)


            overlaps_distance = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
            #计算旋转框iou并除以欧氏距离得到overlaps_distance，距离越小overlaps_distance越大
            overlaps[mask_gt]=rotated_iou_similarity(gt_boxes,pd_boxes)
            overlaps_distance[mask_gt]=overlaps[mask_gt]/Euclidean_distance[mask_gt] 
            min_score_overlaps=overlaps_distance[mask_gt].amin(0)
            max_score_overlaps=overlaps_distance[mask_gt].amax(0)
            overlaps_distance[mask_gt]=(overlaps_distance[mask_gt]-min_score_overlaps+eps)/(max_score_overlaps-min_score_overlaps)

            #align_metric得分已overlaps_distance的得分值为主导
            align_metric = bbox_scores.pow(2) * overlaps_distance.pow(1)
            return align_metric, overlaps_distance
        else:
            if len(gt_boxes)==0:
                align_metric = bbox_scores.pow(2) * overlaps.pow(1)
            else:
                overlaps[mask_gt]=rotated_iou_similarity(gt_boxes,pd_boxes)
                align_metric = bbox_scores.pow(2) * overlaps.pow(1)
            return align_metric, overlaps



    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.
        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.
        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """
        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 5)[target_gt_idx]
        # assigned target scores
        target_labels.clamp(0)
        # target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80) torch.Size([16, 8400, 1])
         # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


