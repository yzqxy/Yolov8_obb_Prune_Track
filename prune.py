# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from models.common import Bottleneck
import numpy as np
import torch
from tqdm import tqdm
import yaml  
from utils.rboxs_utils import poly2hbb, rbox2poly
from utils.prune_utils import gather_bn_weights, obtain_bn_mask
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common_prune import C3Pruned, SPPFPruned, BottleneckPruned,C2fPruned
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader

from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,non_max_suppression_obb,scale_polys,
                           scale_coords, xywh2xyxy, xyxy2xywh,scale_coords2)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from models.yolo import *
from models.yolo_prune import ModelPruned
import numpy as np
from mmcv.ops import box_iou_rotated

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    iou = box_iou_rotated(labels[:, 1:],detections[:, :5])

    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 6]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct



@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        cfg='models/yolov5s.yaml',
        percent=0,
        close_head=None,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
 # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]


    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'HBBmAP@.5', '  HBBmAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    # loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta

        # NMS
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls) # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out): # pred (tensor): (n, [xylsθ, conf, cls])
            # import pdb
            # pdb.set_trace()
            labels = targets[targets[:, 0] == si, 1:7] # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0] # shape (tensor): (h_raw, w_raw)
            seen += 1

 
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Predictions
            if single_cls:
                pred[:, 6] = 0

            scale_coords2(im[si].shape[1:], pred[:, :4], shape, shapes[si][1])

             # Evaluate
            if nl:
                tbox=labels[:, 1:5]
                scale_coords2(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels


                correct = process_batch(pred, labels, iouv)
                if plots:
                    confusion_matrix.process_batch(pred, labels)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)  
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)
            # Save/log
            if save_txt: # just save hbb pred results!
                save_one_txt(pred, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json: # save hbb pred results and poly pred results.
                save_one_json(pred, pred,jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end',pred, pred, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

@torch.no_grad()
def run_prune(data,
        weights=None,  # model.pt path(s)
        cfg = 'models/yolov5s.yaml',
        percent=0,
        close_head=None,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,     
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Data
        data = check_dataset(data)  # check

    # Configure
    model = model.model
    # print(model)
    model.eval()
    # =========================================== prune model ====================================#
    model_list = {}
    ignore_bn_list = []
    #保存的模型不能去除bn层
    for i, layer in model.named_modules():
        if opt.close_head:
            # v8输出头不剪枝则打开，然后对args的传参去除new_channels,不想剪枝那一层往ignore_bn_list添加哪一层
            if isinstance(layer, Detect_v8):
                ignore_bn_list.append(i+".cv2.0.0.bn")
                ignore_bn_list.append(i+".cv2.0.1.bn")
                ignore_bn_list.append(i+".cv2.1.0.bn")
                ignore_bn_list.append(i+".cv2.1.1.bn")
                ignore_bn_list.append(i+".cv2.2.0.bn")
                ignore_bn_list.append(i+".cv2.2.1.bn")

                ignore_bn_list.append(i+".cv3.0.0.bn")
                ignore_bn_list.append(i+".cv3.0.1.bn")
                ignore_bn_list.append(i+".cv3.1.0.bn")
                ignore_bn_list.append(i+".cv3.1.1.bn")
                ignore_bn_list.append(i+".cv3.2.0.bn")
                ignore_bn_list.append(i+".cv3.2.1.bn")

                ignore_bn_list.append(i+".cv4.0.0.bn")
                ignore_bn_list.append(i+".cv4.0.1.bn")
                ignore_bn_list.append(i+".cv4.1.0.bn")
                ignore_bn_list.append(i+".cv4.1.1.bn")
                ignore_bn_list.append(i+".cv4.2.0.bn")
                ignore_bn_list.append(i+".cv4.2.1.bn")              
        if isinstance(layer, torch.nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer

    model_list = {k:v for k,v in model_list.items() if k not in ignore_bn_list}
    prune_conv_list = [layer.replace("bn", "conv") for layer in model_list.keys()]


    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]
    # print("model_list:",model_list)
    # print("bn_weights:",bn_weights)
    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())
    highest_thre = min(highest_thre)
    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(bn_weights)
    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.')
    # assert opt.percent < percent_limit, f"Prune ratio should less than {percent_limit}, otherwise it may cause error!!!"
    # model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * opt.percent)
    thre = sorted_bn[thre_index]
    print('thre',thre)
    print(f'Gamma value that less than {thre:.4f} are set to zero!')
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    remain_num = 0
    modelstate = model.state_dict()
    # ============================== save pruned model config yaml =================================#
    pruned_yaml = {}
    nc = model.model[-1].nc
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    # # Define model
    pruned_yaml["nc"] = model.model[-1].nc
    pruned_yaml["depth_multiple"] = model_yamls["depth_multiple"]
    pruned_yaml["width_multiple"] = model_yamls["width_multiple"]

    # yolov5s
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 3, 2,]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C2fPruned, [128]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C2fPruned, [256]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 6, C2fPruned, [512]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C2fPruned, [1024]],
        [-1, 1, SPPFPruned, [1024, 5]],  # 9
    ]
    pruned_yaml["head"] = [
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C2fPruned, [512, False]],  # 13

        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C2fPruned, [256, False]],  # 17 (P3/8-small)

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 12], 1, Concat, [1]],  # cat head P4
        [-1, 3, C2fPruned, [512, False]],  # 20 (P4/16-medium)

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 9], 1, Concat, [1]],  # cat head P5
        [-1, 3, C2fPruned, [1024, False]],  # 23 (P5/32-large)

        [[15, 18, 21], 1, Detect_v8, [nc]],  # Detect(P3, P4, P5)
    ]


    # ============================================================================== #
    maskbndict = {}

    for bnname, bnlayer in model.named_modules():
  
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            #获取bn_mask并处理为8的整数倍
            mask = obtain_bn_mask(bn_module, thre)  
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size()).cuda()
            maskbndict[bnname] = mask
            
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            bn_module.bias.data.mul_(mask)
            # print("bn_module:", bn_module.bias)
            print(f"|\t{bnname:<25}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
            assert int(mask.sum()) > 0, "Current remaining channel must greater than 0!!! please set prune percent to lower thesh, or you can retrain a more sparse model..."
    print("=" * 94)
    # print('maskbndict',maskbndict)
    #读取剪枝后的模型
    pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    # Compatibility updates
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect_v8, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()

    print('pruned_model_state',len(pruned_model_state.keys()))
    print('modelstate',len(modelstate.keys()))
    # print('pruned_model_state',pruned_model_state.keys())

    assert pruned_model_state.keys() == modelstate.keys()
    # ======================================================================================= #
    changed_state = []
    for ((layername, layer),(pruned_layername, pruned_layer)) in zip(model.named_modules(), pruned_model.named_modules()):
        assert layername == pruned_layername
        # import pdb
        # pdb.set_trace()
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.22"):
            # print('layer',layer)
            # print('pruned_layer',pruned_layer)
            # print('layername',layername)
            convname = layername[:-4]+"bn"
            # print('convname',convname)
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                # print('former',former)
                if isinstance(former, str):
                    #convname model.4.0.m.0.cv1.bn，former model.4.0.cv1.bn
                    #layer Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4]+"bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    # 判断是不是c2f中的bot块，如果是需要split一半，直接获取一半的输入通道数
                    if former[8:]=='cv1.bn' or former[10:]=='cv1.bn' or former[9:]=='cv1.bn' or former[11:]=='cv1.bn':
                        len_indix=int(len(in_idx)/2)
                        in_idx=np.arange(len_indix)
 

                    w = layer.weight.data[:, in_idx, :, :].clone()
                    if len(w.shape) ==3:     # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()

                    pruned_layer.weight.data = w.clone()
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [modelstate[i+".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
  
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:,formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
        
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer,nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()


        if isinstance(layer, nn.Conv2d) and layername.startswith("model.22"):
            convname = layername[:-4]+"bn"
            if convname in from_to_map.keys():               
                former = from_to_map[convname]
                if isinstance(former, str):
                    #convname model.4.0.m.0.cv1.bn，former model.4.0.cv1.bn
                    #layer Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    out_idx =  np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4]+"bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    if len(w.shape) ==3:     # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()
                    # print('w',w.shape)                  
                    pruned_layer.weight.data = w.clone()


    missing = [i for i in pruned_model_state.keys() if i not in changed_state]

    pruned_model.eval()
    pruned_model.names = model.names
    # =============================================================================================== #
    torch.save({"model": model}, "prune/orign_model.pt")
    model = pruned_model
    torch.save({"model":model}, "prune/pruned_model.pt")
    model.cuda().eval()

    finish_pruned_model_state = model.state_dict()

    # print('from_to_map',from_to_map)
    print('finish_pruned_model_state',len(finish_pruned_model_state.keys()))


    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # Dataloader
    if not training:
        # model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, names, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]        

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'HBBmAP@.5', '  HBBmAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    # loss = torch.zeros(4, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls) # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out): # pred (tensor): (n, [xylsθ, conf, cls])
  
            labels = targets[targets[:, 0] == si, 1:7] # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0] # shape (tensor): (h_raw, w_raw)
            seen += 1

 
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Predictions
            if single_cls:
                pred[:, 6] = 0


            scale_coords2(im[si].shape[1:], pred[:, :4], shape, shapes[si][1])

             # Evaluate
            if nl:
                tbox=labels[:, 1:5]
                scale_coords2(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels


                correct = process_batch(pred, labels, iouv)
                if plots:
                    confusion_matrix.process_batch(pred, labels)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)  
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)
            # Save/log
            if save_txt: # just save hbb pred results!
                save_one_txt(pred, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json: # save hbb pred results and poly pred results.
                save_one_json(pred, pred,jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end',pred, pred, path, names, im[si])


        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/voc.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp47/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--close_head', action='store_true')
    parser.add_argument('--percent', type=float, default=0.4, help='prune percentage')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        LOGGER.info(f'test before prune ... ')
        run(**vars(opt))
        LOGGER.info('='*100)
        LOGGER.info('Test after prune ... ')
        run_prune(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
