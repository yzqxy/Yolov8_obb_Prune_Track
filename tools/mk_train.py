import os 
import argparse
def make_dataset(opt):
    data_path=opt.data_path
    with open(data_path+'/train.txt','w') as f:
        for filename1 in os.listdir(data_path):
            if filename1=='images':
                path2=data_path+'/'+filename1
                for filename2 in os.listdir(path2):
                    path3=path2+'/'+filename2
                    if filename2=='train':
                        for filename3 in os.listdir(path3):
                            f.write(path3+'/'+filename3+'\n')
    with open(data_path+'/val.txt','w') as f:
        for filename1 in os.listdir(data_path):
            if filename1=='images':
                path2=data_path+'/'+filename1
                for filename2 in os.listdir(path2):
                    path3=path2+'/'+filename2
                    if filename2=='val':
                        for filename3 in os.listdir(path3):
                            f.write(path3+'/'+filename3+'\n')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/UAV')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    make_dataset(opt)