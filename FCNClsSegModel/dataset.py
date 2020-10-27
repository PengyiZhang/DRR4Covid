import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
import math 
import torch 
import os 

from pathlib import Path
from functools import partial

import argparse
import random

import pickle

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

def default_loader(path):
    image = Image.open(path).convert('L')
    imag_array = np.array(image)
    #imag_array = cv2.equalizeHist(imag_array)
    return Image.fromarray(imag_array).convert('RGB')

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


class XrayClsDataset(data.Dataset):
    def __init__(self, opt, annotations, transform, loader=default_loader):
        # basic slice
        self.annotations = annotations
        if opt.debug:
            self.annotations = self.annotations[:8]
        
        self.transform = transform
        self.loader = loader

        self.opt = opt        


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.annotations) # 
    
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        #print(annotation)
        img_dir = annotation.strip().split(',')[0]
        label = int(annotation.strip().split(',')[1])

        img = self.loader(img_dir)
        img_tensor = self.transform(img)
        label_tensor = torch.tensor([label], dtype=torch.long)
        return img_tensor, label_tensor

class DRRDataset(data.Dataset):
    """
    输出经过transform后的img和配对的mask, 
    img被normalize, 而mask的值域保持不变
    """
    def __init__(self, opt, img_mask_dirs, train=True, loader=default_loader):
        self.annotations = img_mask_dirs
        self.train = train
        self.loader = loader

        if opt.debug:
            self.annotations = self.annotations[:8]
        curr_size = opt.img_size
        min_max_height = opt.crop_sizes
        self.transform_basic = Compose([
            RandomSizedCrop(min_max_height=min_max_height, height=curr_size, width=curr_size, always_apply=True, p=1.0), 
            OneOf([
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5),
                ], p=1.0),
            ])

        self.gt_transform = transforms.Compose([
            lambda img: np.array(img)[np.newaxis, ...],
            #lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.long)
        ])


        self.img_transform = transforms.Compose([
            lambda nd: torch.tensor(nd, dtype=torch.float32),
            lambda nd: nd / 255.0,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.opt = opt
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.annotations) # 
    
    def __getitem__(self, index):
        img_dir, mask_dir = self.annotations[index].strip().split(',')
        img_dir, mask_dir = img_dir.strip(), mask_dir.strip()

        img = np.array(self.loader(img_dir)) 
        mask = cv2.imread(mask_dir, 0) # mask必须为单通道

        if self.opt.seg_augment and self.train:
            img = cv2.resize(img, (self.opt.load_size, self.opt.load_size))
            mask = cv2.resize(mask, (self.opt.load_size, self.opt.load_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (self.opt.img_size, self.opt.img_size))
            mask = cv2.resize(mask, (self.opt.img_size, self.opt.img_size), interpolation=cv2.INTER_NEAREST)            

        if self.opt.seg_augment and self.train:
            au = self.transform_basic(image=img, mask=mask)
            img = au['image']
            mask = au['mask']

    
        img = np.transpose(img, (2,0,1))
        img_tensor = self.img_transform(img)
        mask_tensor = self.gt_transform(mask)

        return img_tensor, mask_tensor


class DRRSegDataset(data.Dataset):
    """
    输出经过transform后的img和配对的mask, 
    img被normalize, 而mask的值域保持不变
    """
    def __init__(self, opt, img_mask_dirs, train=True, loader=default_loader):
        self.annotations = img_mask_dirs
        self.train = train
        self.loader = loader

        if opt.debug:
            self.annotations = self.annotations[:8]
        curr_size = opt.img_size
        min_max_height = opt.crop_sizes

        self.Transforms = {
        "train": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(), # 水平翻转
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(opt.rotation), # 旋转up to 30 degree
            transforms.ToTensor(), # for PIL image, 0-1 range
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "tgt_train": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(), # 水平翻转
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(opt.rotation), # 旋转up to 30 degree
            transforms.ToTensor(), # for PIL image, 0-1 range
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
        self.transform_basic = Compose([
            RandomSizedCrop(min_max_height=min_max_height, height=curr_size, width=curr_size, always_apply=True, p=1.0), 
            OneOf([
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5),
                ], p=1.0),
            ])

        self.gt_transform = transforms.Compose([
            lambda img: np.array(img)[np.newaxis, ...],
            #lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.long)
        ])


        self.img_transform = transforms.Compose([
            lambda nd: torch.tensor(nd, dtype=torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.opt = opt
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.annotations) # 
    
    def __getitem__(self, index):
        img_dir, mask_dir = self.annotations[index].strip().split(',')
        img_dir, mask_dir = img_dir.strip(), mask_dir.strip()

        img = self.loader(img_dir)
        mask = cv2.imread(mask_dir, 0) # mask必须为单通道

        mask = cv2.resize(mask, (self.opt.img_size, self.opt.img_size), interpolation=cv2.INTER_NEAREST)


        img_tensor = self.Transforms['val'](img)
        mask_tensor = self.gt_transform(mask)

        return img_tensor, mask_tensor


class DRRClsDataset(data.Dataset):
    """
    输出经过transform后的img和配对的mask, 
    img被normalize, 而mask的值域保持不变
    """
    def __init__(self, opt, img_mask_dirs, transform, loader=default_loader):
        self.annotations = img_mask_dirs
        if opt.debug:
            self.annotations = self.annotations[:8]


        self.transform = transform
        self.loader = loader

        self.opt = opt
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.annotations) # 
    
    def __getitem__(self, index):
        img_dir, mask_dir = self.annotations[index].strip().split(',')
        img_dir, mask_dir = img_dir.strip(), mask_dir.strip()

        img = self.loader(img_dir) 
        mask = cv2.imread(mask_dir, 0) # mask必须为单通道
        
        label = int(np.sum(mask==self.opt.drr_mask_value_dict["infection"])>0) # 

        img_tensor = self.transform(img)

        label_tensor = torch.tensor([label], dtype=torch.long)

        return img_tensor, label_tensor






def splitXrayClsDataset(args):
    random.seed(args.random_seed)
    root_dir = args.xray_cls_root
    positive_dir = args.xray_cls_positive_dir
    negative_dir = args.xray_cls_negative_dir

    # args.fold 0...4

    positive_names = sorted(os.listdir(os.path.join(root_dir, positive_dir)))
    negative_names = sorted(os.listdir(os.path.join(root_dir, negative_dir)))

    positive_num = len(positive_names)
    

    train_positive_num = positive_num-args.xray_cls_val_num # 119
    val_positive_num = args.xray_cls_val_num

    negative_num = len(negative_names)
    assert negative_num > positive_num

    train_negative_num = train_positive_num
    val_negative_num = val_positive_num

    random.shuffle(positive_names)
    random.shuffle(negative_names)

    train_positive_names = positive_names[:train_positive_num]
    val_positive_names = positive_names[train_positive_num:]

    train_negative_names = negative_names[:train_negative_num]
    val_negative_names = negative_names[train_negative_num:][:val_negative_num]


    train_annotations = [os.path.join(root_dir, positive_dir, train_positive_name) + ",1\n" for train_positive_name in  train_positive_names] + \
                        [os.path.join(root_dir, negative_dir, train_negative_name) + ",0\n" for train_negative_name in  train_negative_names]
    random.shuffle(train_annotations)

    val_annotations = [os.path.join(root_dir, positive_dir, val_positive_name) + ",1\n" for val_positive_name in  val_positive_names] + \
                        [os.path.join(root_dir, negative_dir, val_negative_name) + ",0\n" for val_negative_name in  val_negative_names]
    random.shuffle(val_annotations)

    return train_annotations, val_annotations



def splitXrayClsDataset(args, fold):
    random.seed(args.random_seed)
    root_dir = args.xray_cls_root
    positive_dir = args.xray_cls_positive_dir
    negative_dir = args.xray_cls_negative_dir

    # args.fold 0...4
    splits = load_pickle("split.pkl")

    positive_names = sorted(os.listdir(os.path.join(root_dir, positive_dir)))
    negative_names = sorted(os.listdir(os.path.join(root_dir, negative_dir)))

    positive_num = len(positive_names)
    

    train_positive_num = len(splits[fold]["train"]) #positive_num-args.xray_cls_val_num # 119
    val_positive_num = len(splits[fold]["val"])

    negative_num = len(negative_names)
    assert negative_num > positive_num

    train_negative_num = train_positive_num
    val_negative_num = val_positive_num

    random.shuffle(positive_names)
    random.shuffle(negative_names)

    train_positive_names = splits[fold]["train"]
    val_positive_names = splits[fold]["val"]

    train_negative_names = negative_names[:train_negative_num]
    val_negative_names = negative_names[train_negative_num:][:val_negative_num]


    train_annotations = [os.path.join(root_dir, positive_dir, train_positive_name) + ",1\n" for train_positive_name in  train_positive_names] + \
                        [os.path.join(root_dir, negative_dir, train_negative_name) + ",0\n" for train_negative_name in  train_negative_names]
    random.shuffle(train_annotations)

    val_annotations = [os.path.join(root_dir, positive_dir, val_positive_name) + ",1\n" for val_positive_name in  val_positive_names] + \
                        [os.path.join(root_dir, negative_dir, val_negative_name) + ",0\n" for val_negative_name in  val_negative_names]
    random.shuffle(val_annotations)

    return train_annotations, val_annotations


def splitXrayLungSegDataset(args):
    random.seed(args.random_seed)
    mask_names = sorted(os.listdir(os.path.join(args.xray_lung_seg_root, args.xray_lung_seg_mask_dir)))
    img_names = [mask_name.replace("_mask", "") for mask_name in mask_names]

    total_mask_dirs = [os.path.join(args.xray_lung_seg_root, args.xray_lung_seg_mask_dir, mask_name) for mask_name in mask_names]
    total_img_dirs = [os.path.join(args.xray_lung_seg_root, args.xray_lung_seg_img_dir, img_name) for img_name in img_names]
    total_img_mask_dirs = [_+","+__+"\n" for _, __ in zip(total_img_dirs, total_mask_dirs)]
    random.shuffle(total_img_mask_dirs)

    img_num = len(img_names)
    val_num = args.xray_lung_seg_val_num
    assert img_num > val_num
    train_num = img_num - val_num


    total_train_img_mask_dirs = total_img_mask_dirs[:train_num]
    total_val_img_mask_dirs = total_img_mask_dirs[train_num:]
    
    return total_train_img_mask_dirs, total_val_img_mask_dirs

def get_cls_transforms(opt):
    Transforms = {
        "train": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(), # 水平翻转
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(opt.rotation), # 旋转up to 30 degree
            transforms.ToTensor(), # for PIL image, 0-1 range
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "tgt_train": transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.RandomHorizontalFlip(), # 水平翻转
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(opt.rotation), # 旋转up to 30 degree
            transforms.ToTensor(), # for PIL image, 0-1 range
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return Transforms

def get_drr_img_mask_dirs(args):
    random.seed(args.random_seed)
    total_drr_datasets_weights = args.total_drr_datasets_weights
    total_drr_datasets_thresholds = args.total_drr_datasets_thresholds
    total_drr_datasets_indexes = args.total_drr_datasets_indexes

    selected_drr_datasets_indexes = args.selected_drr_datasets_indexes

    total_mask_dirs = []
    total_img_dirs = []

    for idx in selected_drr_datasets_indexes:
        w0, w1 = total_drr_datasets_weights[idx[0]]
        threshold = total_drr_datasets_thresholds[idx[1]]
        index = total_drr_datasets_indexes[idx[2]]

        root_img_dir = os.path.join(args.drr_root, index, args.drr_folder_prefix+"_%0.1f_%0.1f" % (w0, w1), args.drr_img_dir)
        root_mask_dir = os.path.join(args.drr_root, index, args.drr_folder_prefix+"_%0.1f_%0.1f" % (w0, w1), args.drr_mask_dir, "%0.2f"%threshold)
        img_names = sorted(os.listdir(root_img_dir))
        mask_names = sorted(os.listdir(root_mask_dir))

        assert img_names == mask_names

        total_img_dirs += [os.path.join(root_img_dir, img_name) for img_name in img_names]
        total_mask_dirs += [os.path.join(root_mask_dir, mask_name) for mask_name in mask_names]

    total_num = len(total_img_dirs)

    total_img_mask_dirs = [_+","+__+"\n" for _, __ in zip(total_img_dirs, total_mask_dirs)]

    random.shuffle(total_img_mask_dirs)

    return total_img_mask_dirs

def genDataset(args):
    cls_Transforms = get_cls_transforms(args)
    ################### infection cls
    if args.mode == 0: # x-ray inf cls
        xray_inf_cls_train_annotations, xray_inf_cls_val_annotations = splitXrayClsDataset(args)
        xray_inf_cls_train_dataset = XrayClsDataset(args, xray_inf_cls_train_annotations, cls_Transforms['train'])
        xray_inf_cls_val_dataset = XrayClsDataset(args, xray_inf_cls_val_annotations, cls_Transforms['val'])
        return xray_inf_cls_train_dataset, xray_inf_cls_val_dataset

    if args.mode == 1: # drr inf cls
        drr_total_img_mask_dirs = get_drr_img_mask_dirs(args)
        drr_train_datasets = DRRClsDataset(args, drr_total_img_mask_dirs, cls_Transforms['train'])
        drr_val_datasets = DRRClsDataset(args, drr_total_img_mask_dirs, cls_Transforms['val'])

        _, xray_inf_cls_val_annotations = splitXrayClsDataset(args)
        xray_inf_cls_val_dataset = XrayClsDataset(args, xray_inf_cls_val_annotations, cls_Transforms['val'])
        return drr_train_datasets, xray_inf_cls_val_dataset
    
    if args.mode == 2: # drr inf cls with lmmd
        drr_total_img_mask_dirs = get_drr_img_mask_dirs(args)

        drr_train_datasets = DRRClsDataset(args, drr_total_img_mask_dirs, cls_Transforms['train'])
        drr_val_datasets = DRRClsDataset(args, drr_total_img_mask_dirs, cls_Transforms['val'])

        xray_inf_cls_train_annotations, xray_inf_cls_val_annotations = splitXrayClsDataset(args)
        xray_inf_cls_val_dataset = XrayClsDataset(args, xray_inf_cls_val_annotations, cls_Transforms['val'])
        xray_inf_cls_train_dataset = XrayClsDataset(args, xray_inf_cls_train_annotations, cls_Transforms['tgt_train'])
        return drr_train_datasets, xray_inf_cls_train_dataset, xray_inf_cls_val_dataset

    ################### lung segmentation
    if args.mode == 10: # x-ray lung seg
        xray_train_img_mask_dirs, xray_val_img_mask_dirs = splitXrayLungSegDataset(args)
        xray_train_lung_datasets = DRRDataset(args, xray_train_img_mask_dirs, True)
        xray_val_lung_datasets = DRRDataset(args, xray_val_img_mask_dirs, False)
        return xray_train_lung_datasets, xray_val_lung_datasets
    
    if args.mode == 11: # drr lung seg
        drr_total_img_mask_dirs = get_drr_img_mask_dirs(args)
        drr_train_datasets = DRRDataset(args, drr_total_img_mask_dirs, True)
        drr_val_datasets = DRRDataset(args, drr_total_img_mask_dirs, False)

        xray_train_img_mask_dirs, xray_val_img_mask_dirs = splitXrayLungSegDataset(args)
        xray_val_lung_datasets = DRRDataset(args, xray_val_img_mask_dirs, False)
        
        return drr_train_datasets, xray_val_lung_datasets

    if args.mode == 12: # drr lung seg with lmmd
        drr_total_img_mask_dirs = get_drr_img_mask_dirs(args)
        drr_train_datasets = DRRDataset(args, drr_total_img_mask_dirs, True)
        drr_val_datasets = DRRDataset(args, drr_total_img_mask_dirs, False)

        xray_train_img_mask_dirs, xray_val_img_mask_dirs = splitXrayLungSegDataset(args)
        xray_train_lung_datasets = DRRDataset(args, xray_train_img_mask_dirs, True)
        xray_val_lung_datasets = DRRDataset(args, xray_val_img_mask_dirs, False)

        xray_inf_cls_train_annotations, xray_inf_cls_val_annotations = splitXrayClsDataset(args, args.fold)
        xray_inf_cls_val_dataset = XrayClsDataset(args, xray_inf_cls_val_annotations, cls_Transforms['val'])
        xray_inf_cls_train_dataset = XrayClsDataset(args, xray_inf_cls_train_annotations, cls_Transforms['tgt_train'])
        
        return drr_train_datasets, xray_inf_cls_train_dataset, xray_inf_cls_val_dataset, xray_val_lung_datasets


    ################### lung segmentation, infection segmentation and infection classification
    if args.mode == 21 or args.mode == 22: # drr lung seg
        drr_total_img_mask_dirs = get_drr_img_mask_dirs(args)
        drr_train_datasets = DRRDataset(args, drr_total_img_mask_dirs, True)
        drr_val_datasets = DRRDataset(args, drr_total_img_mask_dirs, False)

        xray_train_img_mask_dirs, xray_val_img_mask_dirs = splitXrayLungSegDataset(args)
        xray_train_lung_datasets = DRRDataset(args, xray_train_img_mask_dirs, True)
        xray_val_lung_datasets = DRRDataset(args, xray_val_img_mask_dirs, False)

        xray_inf_cls_train_annotations, xray_inf_cls_val_annotations = splitXrayClsDataset(args)
        xray_inf_cls_val_dataset = XrayClsDataset(args, xray_inf_cls_val_annotations, cls_Transforms['val'])
        xray_inf_cls_train_dataset = XrayClsDataset(args, xray_inf_cls_train_annotations, cls_Transforms['tgt_train'])

        return drr_train_datasets, drr_val_datasets, xray_train_lung_datasets, xray_val_lung_datasets, xray_inf_cls_train_dataset, xray_inf_cls_val_dataset

def genExtraForEvalDataset(args):
    cls_Transforms = get_cls_transforms(args)
    random.seed(args.random_seed)
    
    root_dir = args.xray_cls_root
    positive_dir = args.xray_cls_positive_dir
    negative_dir = args.xray_cls_negative_dir

    extra_positive_dir = "covid19_512"

    positive_names = sorted(os.listdir(os.path.join(root_dir, positive_dir)))
    extra_positive_names = sorted(os.listdir(os.path.join(root_dir, extra_positive_dir)))
    negative_names = sorted(os.listdir(os.path.join(root_dir, negative_dir)))


    positive_num = len(positive_names)
    extra_positive_num = len(extra_positive_names)
    

    train_positive_num = positive_num-args.xray_cls_val_num # 119
    val_positive_num = args.xray_cls_val_num

    negative_num = len(negative_names)
    assert negative_num > positive_num

    train_negative_num = train_positive_num
    val_negative_num = val_positive_num

    random.shuffle(positive_names)
    random.shuffle(negative_names)


    negative_names = negative_names[train_negative_num+val_negative_num:][:extra_positive_num]



    annotations = [os.path.join(root_dir, extra_positive_dir, train_positive_name) + ",1\n" for train_positive_name in  extra_positive_names] + \
                        [os.path.join(root_dir, negative_dir, train_negative_name) + ",0\n" for train_negative_name in  negative_names]
    random.shuffle(annotations)

    xray_inf_cls_extra_val_dataset = XrayClsDataset(args, annotations, cls_Transforms['val'])



    return xray_inf_cls_extra_val_dataset

def genInfSegDataset(args):
    cls_Transforms = get_cls_transforms(args)
    random.seed(args.random_seed)
    
    root_dir = "../data/target/infection_seg/"
    extra_positive_dir = "images"
    extra_positive_names = sorted(os.listdir(os.path.join(root_dir, extra_positive_dir)))
    extra_positive_num = len(extra_positive_names)

    annotations = [os.path.join(root_dir, extra_positive_dir, train_positive_name) + ",1\n" for train_positive_name in  extra_positive_names]
    random.shuffle(annotations)

    xray_inf_cls_extra_val_dataset = XrayClsDataset(args, annotations, cls_Transforms['val'])



    return xray_inf_cls_extra_val_dataset

if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--config', type=str, default='./cfgs/experiment_seg_lmmd.yaml')


    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    print(args)

    #
    for mode in [0,1,2,10,11,12,21,22]:

        args.mode = mode
        datasets = genDataset(args)
        print("-"*8+"mode %d"%args.mode+"-"*8)
        print([d.annotations for d in datasets])

    


