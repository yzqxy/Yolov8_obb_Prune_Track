# INSTAllation 
## Requirements
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

I have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04/18.04
* CUDA: 10.0/10.1/10.2/11.3

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py39_Torch1.10_cu11.3 python=3.9 -y 
source activate Py39_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the Yolov8_obb_Prune_Track repository.
```
git clone https://github.com/yzqxy/Yolov8_obb_Prune_Track.git
cd Yolov8_obb_Prune_Track
```
e. Install Yolov8_obb_Prune_Track.
```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

f. Install mmrotate.
```python 
pip install mmrotate>=0.3.4
pip install mmcv>=1.6.0
pip install mmcv-full>=1.6.0
```

## Install DOTA_devkit. 
**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**

```
cd Yolov8_obb_Prune_Track/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

