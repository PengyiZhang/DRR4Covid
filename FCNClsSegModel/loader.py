import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import cv2 

# support augmentation
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    Rotate,
    OneOf,
    CLAHE,
    RandomGamma,
    HueSaturationValue,
    IAAAdditiveGaussianNoise, 
    GaussNoise,
    RandomBrightnessContrast,
    IAASharpen, 
    IAAEmboss,
    ElasticTransform,
    CropNonEmptyMaskIfExists,
    RandomCrop,
    RandomSizedCrop,
    RandomScale,
)

## load image
# padding to square shape
# return Image
def default_loader(path):
    image = Image.open(path).convert('L')
    image = np.array(image)
    image = (image*255.0/image.max()).astype(np.uint8) # change to 0-255
    image = Image.fromarray(image).convert('RGB') # Ensure it is square
    return image

def default_loader_forpre(path):
    image = Image.open(path).convert('L')
    image = np.array(image)
    image = (image*255.0/image.max()).astype(np.uint8) # change to 0-255
    # not to rgb, just combine
    image = Image.fromarray(np.stack((image,image,image), -1)).convert('RGB') # Ensure it is square
    return image    

# define the dataset
# designed with following forms：
# annotation_lines : COCO_val2014_000000262148.jpg 1 5 8 14 25 27 37 61
class COVID19Dataset(data.Dataset):
    def __init__(self, root_dir, annotation_lines, class_number, transform=None, loader=default_loader):
        self.annotation_lines = annotation_lines
        self.class_number = class_number
        self.transform = transform
        self.loader = loader
        self.root_dir = root_dir

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip().split(",")
        img = self.loader(os.path.join(self.root_dir,line[0])) # image
        class_id = int(line[1]) 
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor([class_id])

    def __len__(self):
        return len(self.annotation_lines)
    
    def getName(self):
        return "class number = {}".format(self.class_number)


class COVID19SingleDataset(data.Dataset):
    def __init__(self, root_dir, annotation_lines, class_number, transform=None, loader=default_loader):
        self.annotation_lines = annotation_lines
        self.class_number = class_number
        self.transform = transform
        self.loader = loader
        self.root_dir = root_dir
        curr_size = 512
        min_max_height = (curr_size-curr_size//2, curr_size-1)
        self.transform_strong = Compose([
            RandomSizedCrop(min_max_height=min_max_height, height=curr_size, width=curr_size, p=1.0),
            OneOf([
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5),
                ], p=1.0),
            ElasticTransform(alpha=curr_size,sigma=curr_size*0.05,alpha_affine=10,p=1.0)
            ])

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip().split(",")
        img = self.loader(os.path.join(self.root_dir,line[0])) # image
        _parent = self.root_dir.replace("img", "gt")
        cond = cv2.imread(os.path.join(_parent,line[0]), 0)
        
        class_id = int(line[1]) 
        if self.transform is not None:
            img_np = np.array(img)
            au = self.transform_strong(image=img_np, mask=cond)
            #
            img_np = au['image']
            cond = au['mask']

            img = Image.fromarray(img_np)
            img = self.transform(img)

            class_id = np.sum(cond==3)>0

        return img, torch.Tensor([class_id])

    def __len__(self):
        return len(self.annotation_lines)
    
    def getName(self):
        return "class number = {}".format(self.class_number)



class COVID19FuseDataset(data.Dataset):
    def __init__(self, root_dirs, annotation_lines, class_number, transform=None, loader=default_loader):
        self.annotation_lines = annotation_lines
        self.class_number = class_number
        self.transform = transform
        self.loader = loader
        self.root_dirs = root_dirs
        curr_size = 512
        min_max_height = (curr_size-curr_size//2, curr_size-1)
        self.transform_strong = Compose([
            RandomSizedCrop(min_max_height=min_max_height, height=curr_size, width=curr_size, p=1.0),
            OneOf([
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5),
                ], p=1.0),
            ElasticTransform(alpha=curr_size,sigma=curr_size*0.05,alpha_affine=10,p=1.0)
            ])

    def fuse(self, img1, img2):
        """
        # configs == -1, not fuse
        img1[cond==config] = alpha * img1[cond==config] + (1- alpha) * img2[cond==config]
        """
        alpha = random.randint(0,100)/100.0 # 0.3~1.0
        
        img_fuse = alpha * img1 + (1- alpha) * img2

        return img_fuse.astype(np.uint8)

    def __getitem__(self, index):
        line = self.annotation_lines[index].strip().split(",")
        img01 = self.loader(os.path.join(self.root_dirs[0],line[0])) # image
        img02 = self.loader(os.path.join(self.root_dirs[1],line[0])) # image
        img01, img02 = np.array(img01), np.array(img02)
        _parent = self.root_dirs[0].replace("img", "gt")
        cond = cv2.imread(os.path.join(_parent,line[0]), 0)
        # fuse
        img_np = self.fuse(img01, img02) # 
        au = self.transform_strong(image=img_np, mask=cond)

        img_np = au['image']
        cond = au['mask']

        img = Image.fromarray(img_np)

        #class_id = int(line[1]) 
        class_id = np.sum(cond==3)>0
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor([class_id])

    def __len__(self):
        return len(self.annotation_lines)
    
    def getName(self):
        return "class number = {}".format(self.class_number)