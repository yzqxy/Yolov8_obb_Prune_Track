# Getting Started

This page provides basic usage about yolov8-obb. For installation instructions, please see [install.md](./install.md).

# Train a model

**1. Prepare custom dataset files**

1.1 Make sure the labels format is [poly classname diffcult], e.g., You can set **diffcult=0**
```
  x1      y1       x2        y2       x3       y3       x4       y4       classname     diffcult

1686.0   1517.0   1695.0   1511.0   1711.0   1535.0   1700.0   1541.0   large-vehicle      1
```
![image](https://user-images.githubusercontent.com/72599120/159213229-b7c2fc5c-b140-4f10-9af8-2cbc405b0cd3.png)


1.2 Split the dataset. 
```shell
cd Yolov8_obb_Prune_Track
python DOTA_devkit/ImgSplit_multi_process.py
```
or Use the orignal dataset. 
```shell
cd Yolov8_obb_Prune_Track
```

1.3 Make sure your dataset structure same as:
```
└── datasets
    └── your data
          ├── images
              ├── train
                  |────1.jpg
                  |────...
                  └────10000.jpg
              ├── val
                  |────10001.jpg
                  |────...
                  └────11000.jpg
          ├── labelTxt
              ├── train
                    |────1.txt
                    |────...
                    └────10000.txt
              ├── val
                    |────10001.txt
                    |────...
                    └────11000.txt
     └── train.txt
     └── val.txt
```

```shell
python tools/mk_train.py --data_path  data_path
```

**Note:**
* DOTA is a high resolution image dataset, so it needs to be splited before training/testing to get better performance.

**2. Train**

2.1 Train your dataset demo.多卡训练
```shell
python train.py      --data 'data/yolov8obb_demo.yaml'   --hyp 'data/hyps/obb/hyp.finetune_dota.yaml' --cfg models/yolov8n.yaml   --epochs 300   --batch-size 128   --img 640   -- is_use_DP  
```

2.2 单卡训练

```shell
python train.py      --data 'data/yolov8obb_demo.yaml'   --hyp 'data/hyps/obb/hyp.finetune_dota.yaml' --cfg models/yolov8n.yaml   --epochs 300   --batch-size 8   --img 640   --device 1
```

**3. val**
```shell
python val.py --data data/yolov8obb_demo.yaml  --weights runs/train/exp/weights/best.pt --task 'val'  --img 640

```

**4. val_mmrotate**
需要在eval_rotate_PR_V8.py修改你的测试数据集，模型路径等
```shell
python eval_rotate_PR_V8.py 
```


**5. detcet**
```shell
python detect.py --weights  runs/train/exp/weights/best.pt   --source dataset/your datafile/images/val/   --img 640 --device 0 --conf-thres 0.25 --iou-thres 0.2 
```
**6. export**
```shell
python export.py --weights  runs/train/exp/weights/best.pt  --batch 1
```

**7. detcet_save_xml**
测试图片并保存对应的xml文件，可在rolabelimg中打开并进行调整，从而减少标注工作量
```shell
python detcet_save_xml.py --save-xml --xml_save_path_dir your_xml_save_path/  --weights  runs/train/exp/weights/best.pt   --source dataset/your datafile/images/val/   --img 640 --device 0 --conf-thres 0.25 --iou-thres 0.2 
```

**8. yaml file**
除了常规的yolov8n-x的模型结构，还提供了一些轻量化选择，如backbone替换为mobilnetv3，neck替换为AFPN，以及各种注意力机制和transformer结构等等.

# Prune Your Model
**1.Sparity Train**
```shell
#稀疏训练，可选择直接进行稀疏训练，如果直接进行稀疏训练效果不好，可以先进行正常训练到收敛，再进行稀疏训练来微调模型
python train_sparity.py  --st --sr 0.0002  --data 'data/yolov8obb_demo.yaml'   --hyp 'data/hyps/obb/hyp.finetune_dota.yaml' --cfg models/yolov8n.yaml   --epochs 300   --batch-size 8   --img 640   --device 2  --weights yolov8_obb/runs/train/exp/weights/best.pt
```
**2. Prune**
```shell
#剪枝，percent为剪枝比率，如果传入close_head，则不对输出头部分的卷积层进行剪枝。
python prune.py --percent 0.3 --weights runs/train/exp299/weights/last.pt --data data/yolov5obb_demo.yaml --cfg models/yolov8n.yaml --close_head
```
**3. Finetune**
```shell
#微调
python prune_finetune.py --weights prune/pruned_model.pt --data data/yolov5obb_demo.yaml  --epochs 100 --imgsz 640 --batch-size 8
```

# Track
可选参数
video_path：需要预测的跟踪视频读取路径
video_save_path: 跟踪视频预测完的保存路径
video_fps：需要预测的跟踪视频读取帧数
weights: 旋转框检测模型路径
img_save_path：跟踪视频按照video_fps切分后保存图片的路径
track_type：跟踪类型，可选择bytetracker和strongsort
is_track_img：是否存储画有跟踪框的图片
track_img_path：画有跟踪框的图片的存储文件夹路径
is_track_det_img：是否存储画有检测框的图片
track_det_img_path：画有检测框的图片的存储文件夹路径
```shell
#跟踪
python track_predict.py  --video_path --video_fps --weights  --video_save_path
```
