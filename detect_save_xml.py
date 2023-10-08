# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

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
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
from  xml.dom.minidom import Document
from xml.etree import ElementTree as ET

def add_xml_ele(child_ele_name, parent_ele,doc,child_ele_text=None):
    """
    为父节点对象添加子节点对象
    :param child_ele_name:  子节点名称
    :param parent_ele:      父节点对象
    :param child_ele_text:  子节点内容文本
    :return:
    """
    # 创建子节点
    child_ele = doc.createElement(child_ele_name)
    # 父节点添加子节点
    parent_ele.appendChild(child_ele)
    if child_ele_text:
        # 添加文本
        child_ele.appendChild(doc.createTextNode(child_ele_text))

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_xml=False, 
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        xml_save_path_dir=ROOT /'xml_save/img',
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    # device='cpu'
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        # print('pred--',pred.shape)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        print('pred',pred)
        # print('pred:',pred)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # pred_poly = rbox2poly(det[:, :5],det[:, 6:7]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            # print('pred_poly',pred_poly)
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale polys from img_size to im0 size
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                print('pred_poly',pred_poly)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *poly, conf, cls in reversed(det):
                    # print('poly:',*poly)
                    if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        poly = poly.tolist()
                        line = (cls, *poly, conf) if save_conf else (cls, *poly)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_xml:  # Write to file
                        names=['tading',"jyz_r"]

                        # import pdb
                        # pdb.set_trace()
                        # rbox=scale_coords(im.shape[2:], pred, im0.shape)
                        rbox=poly2rbox(pred_poly[:,:8].cpu(),use_pi=True, use_gaussian=False)
                        print('rbox',rbox)
                        
                        img_name=path.split('/')[-1]
                        xml_dir=xml_save_path_dir
                        if not os.path.exists(xml_dir):
                            os.makedirs(xml_dir)

                        xml_name=img_name.replace("jpg", "xml").replace("png", "xml").replace("JPG", "xml")
                        # print('xml_dir+xml_name',xml_dir+'/'+xml_name)
                        # if not os.path.exists(xml_dir+'/'+xml_name):
                        # 创建根元素
                        doc = Document()
                        message_ele = doc.createElement('annotation')
                        # 引用本地XML Schema
                        doc.appendChild(message_ele)
                        # ===== 创建MessageHeader元素 =====
                        add_xml_ele(child_ele_name='folder', parent_ele=message_ele,doc=doc,child_ele_text='img')
                        add_xml_ele(child_ele_name='filename', parent_ele=message_ele, doc=doc,child_ele_text=xml_name)
                        add_xml_ele(child_ele_name='path', parent_ele=message_ele,doc=doc, child_ele_text=xml_name)

                        message_body_ele = doc.createElement('size')
                        add_xml_ele(child_ele_name='width', parent_ele=message_body_ele, doc=doc,child_ele_text=str(im0.shape[1]))
                        add_xml_ele(child_ele_name='height', parent_ele=message_body_ele,doc=doc, child_ele_text=str(im0.shape[0]))
                        add_xml_ele(child_ele_name='depth', parent_ele=message_body_ele,doc=doc, child_ele_text='3')
                        message_ele.appendChild(message_body_ele)

                        add_xml_ele(child_ele_name='segmented', parent_ele=message_ele,doc=doc, child_ele_text='0')

                        for i in range(len(rbox)):
                            print('float(pred[0][i][4].cpu())',float(pred[0][i][4].cpu()))
                            cls_name=names[int(pred[0][i][-1])]                                 
                            object_ele = doc.createElement('object')
                            add_xml_ele(child_ele_name='type', parent_ele=object_ele,doc=doc, child_ele_text='robndbox')
                            add_xml_ele(child_ele_name='name', parent_ele=object_ele, doc=doc,child_ele_text=cls_name)
                            add_xml_ele(child_ele_name='pose', parent_ele=object_ele,doc=doc, child_ele_text='Unspecified')
                            add_xml_ele(child_ele_name='truncated', parent_ele=object_ele,doc=doc, child_ele_text='0')
                            add_xml_ele(child_ele_name='difficult', parent_ele=object_ele,doc=doc, child_ele_text='0')
                            bndbox_ele= doc.createElement('robndbox')
                            add_xml_ele(child_ele_name='cx', parent_ele=bndbox_ele, doc=doc,child_ele_text=str(int(rbox[i][0])))
                            add_xml_ele(child_ele_name='cy', parent_ele=bndbox_ele, doc=doc,child_ele_text=str(int(rbox[i][1])))
                            add_xml_ele(child_ele_name='w', parent_ele=bndbox_ele, doc=doc, child_ele_text=str(int(rbox[i][2])))
                            add_xml_ele(child_ele_name='h', parent_ele=bndbox_ele, doc=doc,child_ele_text=str(int(rbox[i][3])))
                            add_xml_ele(child_ele_name='angle', parent_ele=bndbox_ele, doc=doc,child_ele_text=str(-rbox[i][4]))
                            object_ele.appendChild(bndbox_ele)                         
                            message_ele.appendChild(object_ele)

                            with open(xml_dir+'/'+xml_name, 'w', encoding='utf-8') as f:
                                doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')


                    if save_img or save_crop or view_img:  # Add poly to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.poly_label(poly, label, color=colors(c, True))
                        if save_crop: # Yolov5-obb doesn't support it yet
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            pass

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/yolov5n_DroneVehicle/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DroneVehicle/val/raw/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[840], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-xml', action='store_true', help='save results to *.txt')
    parser.add_argument('--xml_save_path_dir', default='xml_save/img')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
