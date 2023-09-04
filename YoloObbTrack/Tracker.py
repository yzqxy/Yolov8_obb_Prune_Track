'''
Author: yzqxy
'''
from .strong_sort.strong_sort import StrongSORT
from .byte_tracker.byte_tracker import Byte_tracker
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from utils.rboxs_utils import rbox2poly,poly2hbb
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
deepsort = StrongSORT()
byte_tracker = Byte_tracker()



def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def scale_polys_single_img(img1_shape, polys, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings
    
    polys[ [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[ [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[ :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(image, rbox, font,img,ori_img,track_type):

    for rrbox, cls_name, track_id in rbox:
        color = compute_color_for_labels(track_id)
        label = '{}{:d}'.format(cls_name, track_id)
        draw = ImageDraw.Draw(image)
        label = label.encode('utf-8')
        rrbox=rbox2poly(rrbox)
        rrbox=scale_polys_single_img(img,rrbox,ori_img)

        text_origin = np.array([rrbox[0], rrbox[1]], np.int32)      
        draw.polygon(xy=list(rrbox), outline=color, width=2)
        draw.text(text_origin, str(label,'UTF-8'), fill=color, font=font)
        del draw
    return image




def update_tracker( image,pred,PIL_image,img,ori_img,track_type='Byte_tracker'):

    top_label   = np.array(pred[0][:, 6].cpu(), dtype = 'int32')
    rbox = pred[0][:, :5]
    confss =pred[0][:, 5]

    if track_type=='Byte_tracker':
        outputs = byte_tracker.update( rbox, confss, top_label, image)
    else:
        box=poly2hbb(rbox2poly(rbox))
        outputs = deepsort.update( box.cpu(),rbox.cpu(), confss, top_label, image)

    print('outputs',outputs)
    rboxes2draw = []
    current_ids = []
    for value in list(outputs):
        rbox, track_id, cls_, _= value

        rboxes2draw.append((rbox, cls_, track_id))
        current_ids.append(track_id)


    font  = ImageFont.truetype(font='YoloObbTrack/Arial.ttf',
            size=np.floor(3e-2 * PIL_image.size[1] + 0.5).astype('int32'))
    image = draw_boxes(PIL_image, rboxes2draw, font,img,ori_img,track_type)


    return image, rboxes2draw
