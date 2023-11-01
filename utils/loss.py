# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import math
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import numpy as np
from utils.general import LOGGER, check_version
from utils.tal import TaskAlignedAssigner
import torch.nn.functional as F
from utils.loss_cls import *
from utils.loss_bbox import *

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        #边框和角度loss
        self.kld_loss_n = KLDloss(1,fun='log1p')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets



        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        self.no_box=det.no_box
        self.nc = det.nc
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.device=device
        self.varifocal_loss=VarifocalLoss().to(device)
        self.fl=FocalLoss().to(device)
        self.qfl=QFocalLoss().to(device)
        self.eqfl=SoftmaxEQLV2Loss(2).to(device)

        self.reg_max=15
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls,  self.gr, self.hyp, self.autobalance = BCEcls, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets,model_l='l1'):  # predictions, targets, model
        # box, cls, dfl loss
        lcls_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        dfl_loss = torch.zeros(1, device=self.device)
        #网络层输出
        feats = p[1] if isinstance(p, tuple) else p

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        #dfl
        pred_distri,pred_theta,pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no_box, -1) for xi in feats], 2).split(
            (64, 1,self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  #[16, 8400, n]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  #[16, 8400, 64]
        pred_theta = pred_theta.permute(0, 2, 1).contiguous()    #[16, 8400, 1]

        #预测边框通过中心点anchor_points进行边框编码
        pred_bboxes = bbox_decode(anchor_points, pred_distri)  # xywh, (b, h*w, 4),#[16, 8400, 4]
        pred_theta   = (pred_theta.sigmoid()- 0.5) * math.pi
        pred_bboxes=torch.cat((pred_bboxes, pred_theta), -1)
 

        dtype = pred_scores.dtype   #torch.float16
        batch_size = pred_scores.shape[0]  #16
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)#tensor([640., 640.], device='cuda:0', dtype=torch.float16)
        
        #将batch的gt维度进行合并
        targets = preprocess(targets.to(self.device), batch_size, self.device,scale_tensor=imgsz[[1, 0, 1, 0]]) #torch.Size([16, 2, 6])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xyxy torch.Size([16, 2, 1]),torch.Size([16, 2, 5])
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  #torch.Size([16, 2, 1])

        #TAL动态匹配
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        
        target_scores_sum = max(target_scores.sum(), 1)
        target_labels = torch.where(target_scores > 0 , 1, 0)

        #分类vfl loss
        lcls_loss += self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #分类focal loss
        # lcls_loss += self.fl(pred_scores, target_labels.float())   # BCE
        #分类qfocal loss
        # lcls_loss += self.qfl(pred_scores, target_labels.float())  # BCE
        #分类eqfocal loss
        # lcls_loss += self.eqfl(pred_scores, target_labels.float())  # BCE

        if fg_mask.sum():
            #旋转边框值进行下采样，切记不能加入角度
            target_bboxes[:,:,:4] /= stride_tensor
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1).pow(2)
            #边框loss计算
            probloss = probiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            box_loss +=(probloss* weight).sum() / (target_scores_sum*batch_size)

            #边框+角度loss
            # kldloss = self.kld_loss_n(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            # box_loss +=(kldloss* weight).sum() / target_scores_sum

            #DFL loss
            target_ltrb = bbox2dist(anchor_points, target_bboxes[:,:,:4], self.reg_max)
            dfl_loss = df_loss(pred_distri[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
            dfl_loss=dfl_loss.unsqueeze(-1)

        #vfl
        lcls_loss *= self.hyp['cls']
        box_loss *=  self.hyp['box']
        # box_loss *=  0.01
        #qfl
        # lcls_loss *= 10
        # box_loss *= 0.01
        #eqfl
        # lcls_loss *= 0.01
        # box_loss *= 0.01
        dfl_loss *= self.hyp['box']


        # return ( box_loss + lcls_loss ) * batch_size, torch.cat(( box_loss,lcls_loss)).detach()
        return ( box_loss + lcls_loss+dfl_loss ) * batch_size, torch.cat(( box_loss,lcls_loss,dfl_loss)).detach()

   





def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    xy, wh = torch.split(bbox, 2, -1)
    x2y2=(2*xy+wh)/2
    x1y1=(2*xy-wh)/2

    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


TORCH_1_10 = check_version(torch.__version__, '1.10.0')
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        #feats[i],[16, 6, 80, 80],[16, 6, 40, 40],[16, 6, 20, 20]
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # lt, rb,theta = torch.split(distance, [2,2,1], dim)
    # import pdb
    # pdb.set_trace()
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        # c_xy      = c_xy.sigmoid() * 2. - 0.5
        # wh      = (wh.sigmoid() * 2) ** 2 
        
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def bbox_decode(anchor_points, pred_dist):
    device=pred_dist.device
    proj = torch.arange(16, dtype=torch.float, device=device)
    # proj = torch.arange(16, dtype=torch.float, device=device)
    b, a, c = pred_dist.shape  # batch, anchors, channels
    pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=True)

def preprocess( targets, batch_size, device,scale_tensor):

    # import pdb
    # pdb.set_trace()
    if targets.shape[0] == 0:
        out = torch.zeros(batch_size, 0, 6, device=device)
    else:
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 6, device=device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
   

        # out[..., 1:5] = out[..., 1:5].mul_(scale_tensor)
    return out

