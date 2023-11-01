from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F



class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=1.25, gamma=2.0):

        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        # weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
            # loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    # weight).mean(1).sum()
        return loss


   
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, ):
        super().__init__()


    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

        pred_prob = pred.sigmoid()  # prob from logits
        alpha_factor = label * alpha + (1 - label) * (1 - alpha)
        modulating_factor = torch.abs(label - pred_prob) ** gamma
        loss *= alpha_factor * modulating_factor

        return loss.mean(1).sum()   

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()



class SoftmaxEQLV2Loss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLV2Loss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        if label.dim() == 1:
            one_hot = F.one_hot(label, self.num_classes)
        else:
            one_hot = label.clone()
        self.targets = one_hot.detach()

        indicator = indicator / (indicator.sum() + self.eps)
        indicator = (indicator ** self.tau + 1e-9).log()
        cls_score = input + indicator[None, :]
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight
    




class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(min=self.eps)
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight




class BaseLoss(_Loss):
    # do not use syntax like `super(xxx, self).__init__,
    # which will cause infinited recursion while using class decorator`
    def __init__(self,
                 name='base',
                 reduction='none',
                 loss_weight=1.0):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        _Loss.__init__(self, reduction=reduction)
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, input, target, reduction_override=None, normalizer_override=None, **kwargs):
        r"""
        Arguments:
            - input (:obj:`Tensor`)
            - reduction (:obj:`Tensor`)
            - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
            defined in __init__ function

            - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
        """
        reduction = reduction_override if reduction_override else self.reduction
        assert (normalizer_override is None or reduction == 'mean'), \
            f'normalizer is not allowed when reduction is {reduction}'
        loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)
        return loss * self.loss_weight

    def forward(self, input, target, reduction, normalizer=None, **kwargs):
        raise NotImplementedError

class GeneralizedCrossEntropyLoss(BaseLoss):
    def __init__(self,
                 name='generalized_cross_entropy_loss',
                 reduction='none',
                 loss_weight=1.0,
                 activation_type='softmax',
                 ignore_index=-1,):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.activation_type = activation_type
        self.ignore_index = ignore_index

# class SoftMaxFocalLoss(GeneralizedCrossEntropyLoss):
#     def __init__(self,
#                  gamma,
#                  alpha,
#                  num_classes,
#                  init_prior,
#                  name='softmax_focal_loss',
#                  reduction='mean',
#                  loss_weight=1.0,
#                  ignore_index=-1,
#                  ):
#         """
#         Arguments:
#             - name (:obj:`str`): name of the loss function
#             - reduction (:obj:`str`): reduction type, choice of mean, none, sum
#             - loss_weight (:obj:`float`): loss weight
#             - gamma (:obj:`float`): hyparam
#             - alpha (:obj:`float`): hyparam
#             - init_prior (:obj:`float`): init bias initialization
#             - num_classes (:obj:`int`): num_classes total, 81 for coco
#             - ignore index (:obj:`int`): ignore index in label
#         """
#         activation_type = 'softmax'
#         GeneralizedCrossEntropyLoss.__init__(self,
#                                              name=name,
#                                              reduction=reduction,
#                                              loss_weight=loss_weight,
#                                              activation_type=activation_type,
#                                              ignore_index=ignore_index)
#         self.init_prior = init_prior
#         self.num_channels = num_classes
#         self.gamma = gamma
#         self.alpha = alpha
#         assert ignore_index == -1, 'only -1 is allowed for ignore index'

#     def forward(self, input, target, reduction, normalizer=None):
#         """
#         Arguments:
#             - input (FloatTenosor): [[M, N,]C]
#             - target (LongTenosor): [[M, N]]
#         """
#         assert reduction != 'none', 'Not Supported none reduction yet'
#         input = input.reshape(target.numel(), -1)
#         target = target.reshape(-1).int()
#         # normalizer always has a value because the API of `focal loss` need it.
#         normalizer = 1.0 if normalizer is None else normalizer
#         normalizer = torch.Tensor([normalizer]).type_as(input).to(input.device)
#         loss = SoftmaxFocalLossFunction.apply(
#             input, target, normalizer, self.gamma, self.alpha, self.num_channels)
#         return loss
