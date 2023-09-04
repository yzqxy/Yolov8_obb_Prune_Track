# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmcv.utils import print_log
from mmdet.core import average_precision
from terminaltables import AsciiTable

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER,  non_max_suppression_obb)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
import joblib

import json


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 iou_thr=0.3,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 5).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 5). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    
    #一些细长的旋转框的iou可能很微小的扰动就会导致iou急剧下降
    iou_thr=0.5
    # an indicator of ignored gts
    det_bboxes = np.array(det_bboxes)

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp
    
    ious = box_iou_rotated(
        torch.from_numpy(det_bboxes).float(),
        torch.from_numpy(gt_bboxes).float()).numpy()

    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[k, i] = 1
                else:
                    fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """

    cls_dets = [img_res[class_id] for img_res in det_results]

    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])

        else:
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 5)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 5)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_07_metric (bool): Whether to use the voc07 metric.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datasets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """


    print('len(det_results)',len(det_results))
    print('len(annotations)',len(annotations))
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results[0])
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results)  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        # cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
        #     det_results, annotations, i)
        cls_dets=det_results[i]
        cls_gts=annotations[i]

        # for i in range(len(cls_dets)):
        #     print('cls_dets',cls_dets[i])
        #     print('cls_gts',cls_gts[i])
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp



        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
    
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precision = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        if cls_result['precision'].size > 0:
            precision[:, i] = np.array(cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall','precision', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{precision[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


def creat_annotions(label_path):
    #存到json文件
    jdict=[]

    annotions=[]
    #创建多个类别的list
    for i in cls_name_list:
        locals()['list_'+str(i)] = list()

    for label_name in os.listdir(label_path):
        lb_file=label_path + '/' +label_name
        with open(lb_file) as f:
            labels = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l_ = []
            for label in labels:
                cls_id=cls_name_list.index(label[8])
                l_.append(np.concatenate((cls_id, label[:8]), axis=None))
            l = np.array(l_, dtype=np.float32)


        #存储字典数据到列表用于保存到json文件中
        jdict.append(
            {
                'image_id': label_name.split('.')[0]+'jpg',
                'category_id': [int(num) for num in l[:,0].tolist()],
                'bbox_score': [poly2rbox(np.array([num]),use_pi=True, use_gaussian=False).tolist()[0] for num in l[:,1:].tolist()]
                }
        )


        nl = len(l)
        if nl:
            _, i = np.unique(l, axis=0, return_index=True)
            if len(i) < nl:  # duplicate row check
                l = l[i]  # remove duplicates
        else:
            ne = 1  # label empty
            l = np.zeros((0, 9), dtype=np.float32)
        #创建多个类别的list
        for i in cls_name_list:
            locals()['list_every_img_'+str(i)] = list()
        #创建个空数组
        empty_arr=np.empty(shape=(0,5))
        for cls_i in range(len(cls_name_list)):     
            for j in range(nl):
                if int(l[j][0])==cls_i:
                    rboxes = poly2rbox(polys=np.array([l[j, 1:]]),use_pi=True, use_gaussian=False)
                    locals()['list_every_img_'+str(cls_name_list[cls_i])].append(list(rboxes[0]))


        for i in cls_name_list:
            if len(locals()['list_every_img_'+str(i)])==0:
                locals()['list_'+str(i)].append(empty_arr)
            else:
                locals()['list_'+str(i)].append(np.array(locals()['list_every_img_'+str(i)]))
    for i in cls_name_list:
        annotions.append(locals()['list_'+str(i)])

    print('jdict',jdict)
    with open('json_save/v8_pred.json', 'w') as file:
        json.dump(jdict, file)


    return annotions




if __name__ == '__main__':

    pred_list=[]

    device = select_device('0')
    weights='/runs/train/exp/weights/best.pt'
    img_path='/your_datasets/images/val'
    label_path='/your_datasets/labelTxt/val'
    cls_name_list=['tading','jyz_r']

    #读取gt_labels
    annotions=creat_annotions(label_path)
   

    #加载检测模型 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half='False'
    model = DetectMultiBackend(weights, device=device, dnn=False)
    model.model.half() if half else model.model.float()

    #创建多个类别的list
    for i in cls_name_list:
        locals()['det_list_'+str(i)] = list()
    #存储所有类别的检测结果
    det_list=[]
    for img_name in os.listdir(img_path):
        test_img = img_path + '/' + img_name
        print(test_img)
        ori_img=cv2.imread(test_img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        dataset = LoadImages(test_img, img_size=640)
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = model(im)
            # NMS
            pred = non_max_suppression_obb(pred, 0.3, 0.1, multi_label=True, max_det=1000)
            pred=pred[0].cpu().numpy()



            #创建多个类别的list
            for i in cls_name_list:
                locals()['det_list_every_img_'+str(i)] = list()
            #创建个空数组
            empty_arr=np.empty(shape=(0,6))
            for cls_i in range(len(cls_name_list)):     
                #遍历预测结果，将不同类别的输出结果分别保存,如果预测为空，则创建一个空的保持相同维度(0,6)的数组,后面评估vstack需要保持维度一致
                for pred_i in range(len(pred)):
                    if int(pred[pred_i][6])==cls_i:
                        #由于图像的缩放比例，gt是在原图的大小，det是缩放到640以后的结果
                        scale=ori_img.shape[1]/640
                        pred[pred_i,:4]=pred[pred_i,:4]*scale
                        locals()['det_list_every_img_'+str(cls_name_list[cls_i])].append(list(pred[pred_i,:6]))

            for i in cls_name_list:
                if len(locals()['det_list_every_img_'+str(i)])==0:
                    locals()['det_list_'+str(i)].append(empty_arr)
                else:
                    locals()['det_list_'+str(i)].append(np.array(locals()['det_list_every_img_'+str(i)]))

    for i in cls_name_list:
        det_list.append(locals()['det_list_'+str(i)])


    eval_rbbox_map(det_list,annotions)