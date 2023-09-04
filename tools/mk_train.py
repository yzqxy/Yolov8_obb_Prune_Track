import os 

path='/home/yuanzhengqian/yolov8_obb/datasets/UAV/'
with open('/home/yuanzhengqian/yolov8_obb/datasets/train.txt','w') as f:

    for filename1 in os.listdir(path):
        print(filename1)
        if filename1=='images':
            path2=path+'/'+filename1
            for filename2 in os.listdir(path2):
                path3=path2+'/'+filename2
                if filename2=='train':
                    for filename3 in os.listdir(path3):
                        print(filename3)
                        f.write(path3+'/'+filename3+'\n')