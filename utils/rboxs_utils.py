"""
Oriented Bounding Boxes utils
"""
import numpy as np
pi = 3.141592
import cv2
import torch
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形


def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:], 
                           y_sig[:index]], axis=0)

def regular_theta(theta, mode='180', start=-pi/2):
    """
    limit theta ∈ [-pi/2, pi/2)
    """
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def poly2rbox(polys, num_cls_thata=180, radius=6.0, use_pi=False, use_gaussian=False):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
        num_cls_thata (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        use_pi (bool): True θ∈[-pi/2, pi/2) ， False θ∈[0, 180)

    Returns:
        use_gaussian True:
            rboxes (array): 
            csl_labels (array): (num_gts, num_cls_thata)
        elif 
            rboxes (array): (num_gts, [cx cy l s θ]) 
    """
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    rboxes = []
    # print(polys)
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        angle = -angle # θ ∈ [-90， 0]
        theta = angle / 180 * pi # 转为pi制
        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w != max(w, h): 
            w, h = h, w
            theta += pi/2

        # theta = regular_theta(theta) # limit theta ∈ [-pi/2, pi/2)

        angle = (theta * 180 / pi) + 90 # θ ∈ [0， 180)

        # #x,y,w,h->x1,y1,x2,y2
        # x1=x-w/2
        # y1=x-h/2
        # x2=x+w/2
        # y2=y+h/2

        if not use_pi: # 采用angle弧度制 θ ∈ [0， 180)
            rboxes.append([x, y, w, h, angle])
        else: # 采用pi制
            rboxes.append([x, y, w, h, theta])
            # rboxes.append([x1, y1, x2, y2, theta])
        if use_gaussian:
            csl_label = gaussian_label_cpu(label=angle, num_class=num_cls_thata, u=0, sig=radius)
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)

def poly2rbox_2(polys, num_cls_thata=180, radius=6.0, use_pi=False, use_gaussian=False):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
        num_cls_thata (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        use_pi (bool): True θ∈[-pi/2, pi/2) ， False θ∈[0, 180)

    Returns:
        use_gaussian True:
            rboxes (array): 
            csl_labels (array): (num_gts, num_cls_thata)
        elif 
            rboxes (array): (num_gts, [cx cy l s θ]) 
    """
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    rboxes = []
    # print(polys)
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        # print('x,y,w,h',x,y,w,h)
        # print('angle',angle)
        angle = -angle # θ ∈ [-90， 0]
        theta = angle / 180 * pi # 转为pi制

        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w != max(w, h): 
            w, h = h, w
            theta += pi/2
        theta = regular_theta(theta) # limit theta ∈ [-pi/2, pi/2)
        angle = (theta * 180 / pi) + 90 # θ ∈ [0， 180)
        if not use_pi: # 采用angle弧度制 θ ∈ [0， 180)
            rboxes.append([x, y, w, h, angle])

        else: # 采用pi制
            rboxes.append([x, y, w, h, theta])
        if use_gaussian:
            csl_label = gaussian_label_cpu(label=angle, num_class=num_cls_thata, u=0, sig=radius)
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)



# def rbox2poly(rboxes):
#     """
#     Trans rbox format to poly format.
#     Args:
#         rboxes (array): (num_gts, [cx cy l s θ]) θ∈(0, 180]

#     Returns:
#         polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
#     """
#     assert rboxes.shape[-1] == 5
#     polys = []
#     for rbox in rboxes:
#         x, y, w, h, theta = rbox
#         if theta > 90 and theta <= 180: # longedge format -> opencv format
#             w, h = h, w
#             theta -= 90
#         if theta <= 0 or theta > 90:
#             print("cv2.minAreaRect occurs some error. θ isn't in range(0, 90]. The longedge format is: ", rbox)

#         poly = cv2.boxPoints(((x, y), (w, h), theta)).reshape(-1)  
#         polys.append(poly)
#     return np.array(polys)

def rbox2poly2(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:,:, :2], obboxes[:,:, 2:3], obboxes[:,:, 3:4], obboxes[:,:, 4:5]

        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point2, point3, point4,point1), dim=-1).reshape(*order, 4,2)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point2, point3, point4,point1], axis=-1).reshape(*order, 8)


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]

        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def rbox2poly3(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:2], obboxes[ 2:3], obboxes[ 3:4], obboxes[ 4:5]

        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def rbox2poly_gt(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        x1,y1, x2, y2, theta = obboxes[:, :1], obboxes[:, 1:2],obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
  
        c_x= (x1 + x2) / 2
        c_y= (y1 + y2) / 2
        w = x2 - x1
        h=  y2-y1
        center=torch.cat((c_x,c_y),dim=1)

        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def rbox2poly_tal(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:,:, :2], obboxes[:,:, 2:3], obboxes[:,:, 3:4], obboxes[:,:, 4:5]
        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1) 
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes

def poly2hbb_tal(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):

        x = polys[:,:, 0::2] # (num, 4) 
        y = polys[:,:, 1::2]
        x_max = torch.amax(x, dim=2) # (num)
        x_min = torch.amin(x, dim=2)
        y_max = torch.amax(y, dim=2)
        y_min = torch.amin(y, dim=2)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1).view(polys.shape[0],polys.shape[1],4)
    else:
        x = polys[:,:, 0::2] # (num, 4) 
        y = polys[:,:, 1::2]
        x_max = np.amax(x, axis=2) # (num)
        x_min = np.amin(x, axis=2) 
        y_max = np.amax(y, axis=2)
        y_min = np.amin(y, axis=2)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=2)
    return hbboxes


def poly_filter(polys, h, w): 
    """
    Filter the poly labels which is out of the image.
    Args:
        polys (array): (num, 8)

    Return：
        keep_masks (array): (num)
    """
    x = polys[:, 0::2] # (num, 4) 
    y = polys[:, 1::2]
    x_max = np.amax(x, axis=1) # (num)
    x_min = np.amin(x, axis=1) 
    y_max = np.amax(y, axis=1)
    y_min = np.amin(y, axis=1)
    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
    keep_masks = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) 
    return keep_masks

def box2corners(box):
    """convert box coordinate to corners
    Args:
        box (Tensor): (B, N, 5) with (x, y, w, h, alpha) angle is in [0, 90)
    Returns:
        corners (Tensor): (B, N, 4, 2) with (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    B = box.shape[0]

    x, y, w, h, alpha = torch.split(box, [1,1,1,1,1], 2)
    device = x.device
    x4 = torch.tensor(
        [0.5, 0.5, -0.5, -0.5], dtype=torch.float32).reshape(
            (1, 1, 4)).to(device)  # (1,1,4)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.tensor(
        [-0.5, 0.5, 0.5, -0.5], dtype=torch.float32).reshape((1, 1, 4)).to(device)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.concat([cos, sin], dim=-1)
    row2 = torch.concat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.reshape([-1, 4, 2]), rot_T.reshape([-1, 2, 2]))
    rotated = rotated.reshape([B, -1, 4, 2])  # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def iou_eight(bbox, candidates,eps=1e-7):


    a = bbox.cpu().numpy().reshape((4, 2))  # 四边形二维坐标表示,四行两列
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    #print('凸包值',Polygon(a).convex_hull)  # 可以打印看看是不是这样子


    b = candidates.cpu().numpy().reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area

            # print(union_area)
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