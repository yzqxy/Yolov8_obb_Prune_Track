import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.common_prune import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

TORCH_1_10 = check_version(torch.__version__, '1.10.0')
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class Detect_v8(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    dynamic = False  # force grid reconstruction
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    shape = None
    export = False  # export mode
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        #dfl
        self.no_box = nc + self.reg_max * 4 +1   # number of outputs per anchor

        # self.no_box = nc + 4 +1   # number of outputs per anchor

        self.nl =  len(ch) # number of detection layers
        self.na = 3  # number of anchors
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.theta=1
        # c2, c3,c4 = max((16, ch[0] // 4,5)), max(ch[0], self.nc),max(ch[0],0)   # channels

        c2, c3,c4 = max((16, ch[0] // 4,self.reg_max * 4)), max(ch[0], self.nc),max(ch[0],1)   # channels

        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2,self.reg_max * 4, 1)) for x in ch)
        # self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.theta, 1)) for x in ch)
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]),self.cv4[i](x[i]), self.cv3[i](x[i]) ), 1)
            # print(' x[i]', x[i].shape)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # box, cls = torch.cat([xi.view(shape[0], self.no_box, -1) for xi in x], 2).split((5, self.nc), 1)
        # box,theta, cls = torch.cat([xi.view(shape[0], self.no_box, -1) for xi in x], 2).split((4, self.theta ,self.nc), 1)
        # dbox = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        #dfl_box
        box,theta, cls = torch.cat([xi.view(shape[0], self.no_box, -1) for xi in x], 2).split((self.reg_max * 4, self.theta ,self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        y = torch.cat((dbox,theta, cls.sigmoid()), 1)
 
        return y if self.export else (y, x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class ModelPruned(nn.Module):
    def __init__(self,maskbndict, cfg='yolov8n.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super().__init__()
        self.maskbndict = maskbndict
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save,self.from_to_map  = parse_pruned_model(self.maskbndict,deepcopy(self.yaml), ch=[ch])  # model, savelistW

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # import pdb
        # pdb.set_trace()
        # Build strides, anchors
        m = self.model[-1]  # Detect()


        # if isinstance(m, Detect_v8):
        s = 256  # 2x min stride
        m.inplace = self.inplace
        forward = lambda x: self.forward(x)
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
        self.stride = m.stride
        m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        """
        Args:
            x (tensor): (b, 3, height, width), RGB

        Return：
            if not augment:
                x (list[P3_out, ...]): tensor.Size(b, self.na, h_i, w_i, c), self.na means the number of anchors scales
            else:
                
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train


    def _forward_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False

        Returns:
            (torch.tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                LOGGER.info('visualize feature not yet supported')
                # TODO: feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)




def parse_pruned_model(maskbndict, d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    print('gd',gd)
    print('gw',gw)
    fromlayer = []
    from_to_map = {}
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # print('d---',d)
    
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print('ch',ch)
        # print('n',n)
        # print('m',m)
        # print('f',f)
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m in [Conv]:
            named_m_bn = named_m_base + ".bn"
            bnc = int(maskbndict[named_m_bn].sum())
            c1, c2 = ch[f], bnc
            args = [c1, c2, *args[1:]]
            layertmp = named_m_bn
            if i>0:
                from_to_map[layertmp] = fromlayer[f]         
            fromlayer.append(named_m_bn)

        elif m in [C2fPruned]:
            # #model.4和6的cf2为n=2层数更多需要注意规则
            # if (named_m_base == 'model.4' and gw==0.25) or (named_m_base == 'model.6' and gw==0.25):
            #     args_list=[]

            #     for q in range(n):
            #         named_m_cv1_bn = named_m_base + ".{}.cv1.bn".format(q)
            #         named_m_cv2_bn = named_m_base + ".{}.cv2.bn".format(q)
                   
            #         from_to_map[named_m_cv2_bn] = fromlayer[f]
                    
            #         cv1in = ch[f]

            #         cv1out = int(maskbndict[named_m_cv1_bn].sum())
            #         cv2out = int(maskbndict[named_m_cv2_bn].sum())
                    
            #         c3fromlayer = [named_m_cv1_bn]

            #         named_m_bottle_cv1_bn = named_m_base + ".{}.m.0.cv1.bn".format(q)
            #         named_m_bottle_cv2_bn = named_m_base + ".{}.m.0.cv2.bn".format(q)
  

            #         bottle_cv1out = int(maskbndict[named_m_bottle_cv1_bn].sum())
            #         bottle_cv2out = int(maskbndict[named_m_bottle_cv2_bn].sum())

            #         bottle_args = []
            #         #Bottleneck_C2f的传参，int(cv1in/2)是该模块的split操作
            #         bottle_args.append([int(cv1out/2), bottle_cv1out, bottle_cv2out])
            #         from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[0]
            #         from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            #         c3fromlayer.append(named_m_bottle_cv2_bn)

            #         from_to_map[named_m_cv2_bn] = [c3fromlayer[-1], named_m_cv1_bn]
            #         if q ==0:
            #             args = [cv1in, cv1out, cv2out, 1, args[-1]]
            #             args.insert(3, bottle_args)
            #             args_list.append(args)
            #             from_to_map[named_m_cv1_bn] = fromlayer[f]
            #         else:
            #             args = [args_list[0][2], cv1out, cv2out, 1, args[-1]]
            #             args.insert(3, bottle_args)
            #             args_list.append(args)
            #             from_to_map[named_m_cv1_bn] = named_m_base + ".0.cv2.bn"

            #     c2 = cv2out
            #     fromlayer.append(named_m_cv2_bn)    
            #     print('args_list',args_list)
            #     n = 2
            # else :
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            fromlayer.append(named_m_cv2_bn)

            cv1in = ch[f]
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())

            args = [cv1in, cv1out, cv2out, n, args[-1]]
            bottle_args = []
            c3fromlayer = [named_m_cv1_bn]

            for p in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(p)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(p)


                bottle_cv1out = int(maskbndict[named_m_bottle_cv1_bn].sum())
                bottle_cv2out = int(maskbndict[named_m_bottle_cv2_bn].sum())

                bottle_args.append([int(cv1out/2), bottle_cv1out, bottle_cv2out])
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
            if n>1:
                bottle_args[1][0]=bottle_args[0][2]                    
            args.insert(3, bottle_args)

            c2 = cv2out
            n = 1
            from_to_map[named_m_cv2_bn] = c3fromlayer

        elif m in [SPPFPruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            cv1in = ch[f]


            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn]*4
            fromlayer.append(named_m_cv2_bn)
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:   
            c2 = sum(ch[x] for x in f)
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)

        elif m is Detect_v8:
            from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
            from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
            from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            if isinstance(f, int):
                c2 = ch[f]
                fromtmp = fromlayer[-1]
                fromlayer.append(fromtmp)
            else:
    
                # #yolov8输出头接了连续3个输出卷积层  ,判断m is Detect_v8时并不是true，目前没找到原因。
                #记录该层的输入bn层，即上一层的输出通道数
                from_to_map[named_m_base + ".cv2.0.0.bn"] = fromlayer[f[0]]
                from_to_map[named_m_base + ".cv2.0.1.bn"] = named_m_base + ".cv2.0.0.bn"
                from_to_map[named_m_base + ".cv2.0.2.bn"] = named_m_base + ".cv2.0.1.bn"
                from_to_map[named_m_base + ".cv2.1.0.bn"] = fromlayer[f[1]]
                from_to_map[named_m_base + ".cv2.1.1.bn"] = named_m_base + ".cv2.1.0.bn"
                from_to_map[named_m_base + ".cv2.1.2.bn"] = named_m_base + ".cv2.1.1.bn"
                from_to_map[named_m_base + ".cv2.2.0.bn"] = fromlayer[f[2]]
                from_to_map[named_m_base + ".cv2.2.1.bn"] = named_m_base + ".cv2.2.0.bn"
                from_to_map[named_m_base + ".cv2.2.2.bn"] = named_m_base + ".cv2.2.1.bn"

                from_to_map[named_m_base + ".cv3.0.0.bn"] = fromlayer[f[0]]
                from_to_map[named_m_base + ".cv3.0.1.bn"] = named_m_base + ".cv3.0.0.bn"
                from_to_map[named_m_base + ".cv3.0.2.bn"] = named_m_base + ".cv3.0.1.bn"
                from_to_map[named_m_base + ".cv3.1.0.bn"] = fromlayer[f[1]]
                from_to_map[named_m_base + ".cv3.1.1.bn"] = named_m_base + ".cv3.1.0.bn"
                from_to_map[named_m_base + ".cv3.1.2.bn"] = named_m_base + ".cv3.1.1.bn"
                from_to_map[named_m_base + ".cv3.2.0.bn"] = fromlayer[f[2]]
                from_to_map[named_m_base + ".cv3.2.1.bn"] = named_m_base + ".cv3.2.0.bn"
                from_to_map[named_m_base + ".cv3.2.2.bn"] = named_m_base + ".cv3.2.1.bn"

                from_to_map[named_m_base + ".cv4.0.0.bn"] = fromlayer[f[0]]
                from_to_map[named_m_base + ".cv4.0.1.bn"] = named_m_base + ".cv4.0.0.bn"
                from_to_map[named_m_base + ".cv4.0.2.bn"] = named_m_base + ".cv4.0.1.bn"
                from_to_map[named_m_base + ".cv4.1.0.bn"] = fromlayer[f[1]]
                from_to_map[named_m_base + ".cv4.1.1.bn"] = named_m_base + ".cv4.1.0.bn"
                from_to_map[named_m_base + ".cv4.1.2.bn"] = named_m_base + ".cv4.1.1.bn"
                from_to_map[named_m_base + ".cv4.2.0.bn"] = fromlayer[f[2]]
                from_to_map[named_m_base + ".cv4.2.1.bn"] = named_m_base + ".cv4.2.0.bn"
                from_to_map[named_m_base + ".cv4.2.2.bn"] = named_m_base + ".cv4.2.1.bn"


                #保存剪枝后的新通道数，如果不剪枝v8输出头则关闭
                cv2_out1 = int(maskbndict[named_m_base + ".cv2.0.0.bn"].sum())
                cv2_out2 = int(maskbndict[named_m_base + ".cv2.0.1.bn"].sum())
                cv2_out3 = int(maskbndict[named_m_base + ".cv2.1.0.bn"].sum())
                cv2_out4 = int(maskbndict[named_m_base + ".cv2.1.1.bn"].sum())
                cv2_out5 = int(maskbndict[named_m_base + ".cv2.2.0.bn"].sum())
                cv2_out6 = int(maskbndict[named_m_base + ".cv2.2.1.bn"].sum())
                cv2_list=[[cv2_out1,cv2_out2],[cv2_out3,cv2_out4],[cv2_out5,cv2_out6]]

                cv3_out1 = int(maskbndict[named_m_base + ".cv3.0.0.bn"].sum())
                cv3_out2 = int(maskbndict[named_m_base + ".cv3.0.1.bn"].sum())
                cv3_out3 = int(maskbndict[named_m_base + ".cv3.1.0.bn"].sum())
                cv3_out4 = int(maskbndict[named_m_base + ".cv3.1.1.bn"].sum())
                cv3_out5 = int(maskbndict[named_m_base + ".cv3.2.0.bn"].sum())
                cv3_out6 = int(maskbndict[named_m_base + ".cv3.2.1.bn"].sum())
                cv3_list=[[cv3_out1,cv3_out2],[cv3_out3,cv3_out4],[cv3_out5,cv3_out6]]

                cv4_out1 = int(maskbndict[named_m_base + ".cv4.0.0.bn"].sum())
                cv4_out2 = int(maskbndict[named_m_base + ".cv4.0.1.bn"].sum())
                cv4_out3 = int(maskbndict[named_m_base + ".cv4.1.0.bn"].sum())
                cv4_out4 = int(maskbndict[named_m_base + ".cv4.1.1.bn"].sum())
                cv4_out5 = int(maskbndict[named_m_base + ".cv4.2.0.bn"].sum())
                cv4_out6 = int(maskbndict[named_m_base + ".cv4.2.1.bn"].sum())
                cv4_list=[[cv4_out1,cv4_out2],[cv4_out3,cv4_out4],[cv4_out5,cv4_out6]]

                #传入yolo.py中的Detect_v8函数的参数，如果不剪枝输出头，则new_channel通道数为0
                args=[2,[ch[f[0]],ch[f[1]],ch[f[2]]],[cv2_list,cv3_list,cv4_list]]
                #不剪枝则不传入new_channels参数
                # args=[2,[ch[f[0]],ch[f[1]],ch[f[2]]]]

            


        m_ = nn.Sequential(*(m(*args_list[i_p]) for i_p in range(n))) if n > 1 else m(*args)  # module
        # print('m_',m_)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        # print('ch',ch)
        # print('from_to_map',from_to_map)
    return nn.Sequential(*layers), sorted(save), from_to_map

