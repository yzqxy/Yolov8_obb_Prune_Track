import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDloss(nn.Module):

    def __init__(self, taf=1.0, fun="sqrt"):
        super(KLDloss, self).__init__()
        self.fun = fun
        self.taf = taf
        self.pi = 3.141592
    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        #assert pred.shape[0] == target.shape[0]
        # print('pred',pred.shape)
        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        
        pre_angle_radian = pred[:, 4]
        targrt_angle_radian = target[:, 4]


        # pre_angle_radian =  self.pi *(((pred[:, 4] * 180 / self.pi ) + 90)/180)
        # targrt_angle_radian = self.pi *(((target[:, 4] * 180 / self.pi ) + 90)/180)

        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        

        if self.fun == "sqrt":
            kld = kld.clamp(1e-7).sqrt()
        elif self.fun == "log1p":
            kld = torch.log1p(kld.clamp(1e-7))
        else:
            pass

        kld_loss = 1 - 1 / (self.taf + kld)

        return kld_loss
    

def gbb_form(boxes):
    xy, wh, angle = torch.split(boxes, [2, 2, 1], dim=-1)
    return torch.concat([xy, wh.pow(2) / 12., angle], dim=-1)


def rotated_form(a_, b_, angles):
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    a = a_ * torch.pow(cos_a, 2) + b_ * torch.pow(sin_a, 2)
    b = a_ * torch.pow(sin_a, 2) + b_ * torch.pow(cos_a, 2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c


def probiou_loss(pred, target,  mode='l1'):
    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """

    eps=1e-3
    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    beta=1.0 / 9.0
    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,
                                     0], gbboxes1[:,
                                                  1], gbboxes1[:,
                                                               2], gbboxes1[:,
                                                                            3], gbboxes1[:,
                                                                                         4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,
                                     0], gbboxes2[:,
                                                  1], gbboxes2[:,
                                                               2], gbboxes2[:,
                                                                            3], gbboxes2[:,
                                                                                         4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = 0.25 * ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) + \
         0.5 * ((c1+c2)*(x2-x1)*(y1-y2))
    t2 = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(F.relu(t3_)) + eps))

    B_d = (t1 / t2) + t3
    # B_d = t1 + t2 + t3

    B_d = torch.clip(B_d, min=eps, max=100.0)
    l1 = torch.sqrt(1.0 - torch.exp(-B_d) + eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1
    if mode == 'l2':
        probiou = l2

    return probiou
    # return probiou+xy_loss



def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[:, :2]
    wh = xywhr[:, 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[:, 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """

    pred_decode=pred_decode.float()
    targets_decode=targets_decode.float()
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)
    Sigma_p=Sigma_p.float()
    Sigma_t=Sigma_t.float()
    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()

    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 *Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss

def df_loss(pred_dist, target):
    # Return sum of left and right DFL losses
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    tl = target.long()  # target left
    tr = tl + 1  # target right
    wl = tr - target  # weight left
    wr = 1 - wl  # weight right
    return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
            F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)
