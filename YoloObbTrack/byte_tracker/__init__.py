from .byte_tracker import Byte_tracker


__all__ = ['Byte_tracker', 'build_tracker']


def build_tracker(cfg, use_cuda):
    return Byte_tracker(cfg.STRONGSORT.REID_CKPT, 
                max_dist=cfg.STRONGSORT.MAX_DIST, min_confidence=cfg.STRONGSORT.MIN_CONFIDENCE, 
                nms_max_overlap=cfg.STRONGSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE, 
                max_age=cfg.STRONGSORT.MAX_AGE, n_init=cfg.STRONGSORT.N_INIT, nn_budget=cfg.STRONGSORT.NN_BUDGET, use_cuda=use_cuda)
