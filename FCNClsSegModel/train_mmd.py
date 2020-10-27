# coding: utf-8
"""
"""
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import copy
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from visual_confuse_matrix import make_confusion_matrix
from dataset import genDataset
from model import SegClsModule

from sklearn.metrics import cohen_kappa_score
import argparse
import logging
import os
import sys
import torchvision.transforms as transforms
import cv2 
import numpy as np 
import math

import random
import yaml
from pathlib import Path 

from loss import Weighted_Jaccard_loss
from utils import dice_coef, probs2one_hot


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """terminal and log file
    name: application information
    save_dir: log dir
    distributed_rank: only host 0 can generate log
    filename: log file name
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_visible_gpu(gpu_idex):
    """
    to control which gpu is visible for CUDA user
    
    set_visible_gpu(1)
    print(os.environ["CUDA_DEVICE_ORDER"])
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu_idex)


def get_results(val_labels, val_outs, save_cf_png_dir, save_metric_dir):

    CM = confusion_matrix(val_labels, val_outs)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    categories = ['0', '1']
    make_confusion_matrix(CM, 
                        group_names=labels,
                        categories=categories, 
                        cmap='Blues',save_dir=save_cf_png_dir)
    #make_confusion_matrix(CM, figsize=(8,6), cbar=False)


    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    result_str = "Sensitivity=%.3f, Specificity=%.3f, PPV=%.3f, NPV=%.3f, FPR=%.3f, FNR=%.3f, FDR=%.3f, ACC=%.3f\n" % (TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC)
    
    save_dir = save_metric_dir
    with open(save_dir, "a+") as f:
        f.writelines([result_str])
    return result_str

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]
        
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, log_dir="./log/", scheduler=None, writer=None, logger=None, opt=None):
    print(opt)
    since = time.time()
    val_acc_history = []

    best_acc = 0.0

    batch_size = dataloaders['train'].batch_size
    print_step = 5 # print info per 10 batches

    val_losses = []

    tgt_cls_train_iter = iter(dataloaders['tgt_cls_train'])


    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        learning_rate = get_learning_rate(optimizer)
        writer.add_scalar("lr", learning_rate, epoch)

        epoch_val_preds = []
        epoch_val_y = []

        epoch_train_preds = []
        epoch_train_y = []

        # Each epoch has a training and validation phase

            
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloaders["train"], 0):

            inputs = inputs.to(device)

            # adjust labels
            labels[labels==opt.drr_mask_value_dict["lung"]] = 1
            labels[labels==opt.drr_mask_value_dict["infection"]] = 2

            labels = labels[:,-1].to(device)
            tag_labels = ((labels == 2).sum(-1).sum(-1) > 0).type(torch.long).to(device) # batch_size, 1

            c_labels = tag_labels if opt.do_cls_mmd else None
            s_labels = labels if opt.do_seg_mmd else None

            if opt.do_cls_mmd or opt.do_seg_mmd:
                # tgt_cls
                try:
                    tgt_inputs, _ = tgt_cls_train_iter.next()
                except StopIteration:
                    tgt_cls_train_iter = iter(dataloaders['tgt_cls_train'])
                    tgt_inputs, _ = tgt_cls_train_iter.next()

                tgt_inputs = tgt_inputs.to(device)
            else:
                tgt_inputs = None 
            
            # zero the parameter gradients
            optimizer.zero_grad()
            model.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                src_cls_logits, loss_cls_lmmd, src_seg_logits, loss_seg_lmmd, _ = model(inputs, tgt_img=tgt_inputs, c_label=c_labels, s_label=s_labels)
                
                lambd = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1

                if opt.do_cls and opt.do_cls_mmd:
                    loss_cls_lmmd = lambd * loss_cls_lmmd * opt.lambda_cls_mmd
                    loss_cls_lmmd_item = loss_cls_lmmd.item()
                else:
                    loss_cls_lmmd = 0
                    loss_cls_lmmd_item = 0

                if opt.do_seg and opt.do_seg_mmd:
                    loss_seg_lmmd = lambd * loss_seg_lmmd * opt.lmabda_seg_mmd
                    loss_seg_lmmd_item = loss_seg_lmmd.item()
                else:
                    loss_seg_lmmd = 0
                    loss_seg_lmmd_item = 0

                if opt.do_seg:
                    loss_seg = criterion(labels, src_seg_logits, class_weights=opt.seg_class_weights) * opt.lambda_seg
                    loss_seg_item = loss_seg.item()
                else:
                    loss_seg = 0
                    loss_seg_item = 0

                if opt.do_cls:
                    loss_cls = F.cross_entropy(src_cls_logits, tag_labels) * opt.lambda_cls
                    loss_cls_item = loss_cls.item()
                else:
                    loss_cls = 0
                    loss_cls_item = 0

                loss = loss_seg + loss_cls + loss_seg_lmmd + loss_cls_lmmd
                loss_item = loss.item()

                loss.backward()
                optimizer.step()

            # statistics
            if batch_idx % print_step == 0: # print info
                print_loss = running_loss / ((batch_idx+1)*batch_size)
                logger.info("Train E{:>03} B{:>05} LR:{:.8f} Loss: {:.4f} LSeg: {:.4f} SegMmd: {:.4f} LCls: {:.4f} ClsMmd: {:.4f}".format(epoch, batch_idx, learning_rate, loss_item, loss_seg_item, loss_seg_lmmd_item, loss_cls_item, loss_cls_lmmd_item))
                
        scheduler.step()
        weight_path = os.path.join(log_dir, "latest.pth")
        torch.save(model.state_dict(), weight_path)

        # if ((epoch+1) % opt.eval_times == 0 or epoch+1 == num_epochs) and opt.do_seg:
            
        #     # eval lung segmentation
        #     logger.info("-"*8+"eval lung segmentation"+"-"*8)

        #     model.eval()
        #     all_dices = []

        #     for batch_idx, (inputs, labels) in enumerate(dataloaders["tgt_lung_seg_val"], 0):
        #         annotation = dataloaders["tgt_lung_seg_val"].dataset.annotations[batch_idx]
        #         img_dir = annotation.strip().split(',')[0]
        #         img_name = Path(img_dir).name                
                
                
        #         inputs = inputs.to(device)
        #         # adjust labels
                
        #         labels[labels==opt.xray_mask_value_dict["lung"]] = 1
                
        #         labels = labels[:,-1].to(device)
        #         labels = torch.stack([labels == c for c in range(2)], dim=1)

                
                
        #         with torch.set_grad_enabled(False):
        #             _, _, seg_logits, _, _ = model(inputs)
        #             seg_probs = torch.softmax(seg_logits, dim=1)
        #             predicted_mask = probs2one_hot(seg_probs.detach())

        #             # change the infection to Lung
        #             predicted_mask_lung = predicted_mask[:,:-1]
        #             predicted_mask_lung[:,-1] += predicted_mask[:,-1]
        #             dices = dice_coef(predicted_mask_lung, labels.detach().type_as(predicted_mask)).cpu().numpy()


        #             all_dices.append(dices) # [(B,C)]

        #             predicted_mask_lung = predicted_mask_lung.squeeze().cpu().numpy() # 3xwxh
        #             mask_inone = (np.zeros_like(predicted_mask_lung[0])+predicted_mask_lung[1]*255).astype(np.uint8)

        #             # save dir:
        #             save_dir = os.path.join(opt.logs, "tgt_lung_seg_val", "ep%03d"%epoch)
        #             # 
        #             if not os.path.exists(save_dir):
        #                 os.makedirs(save_dir)
                    
        #             cv2.imwrite(os.path.join(save_dir, img_name), mask_inone)


        #         avg_dice = np.mean(np.concatenate(all_dices, 0), 0) #


        #         logger.info("tgt_lung_seg_val:EP%03d,[%d/%d],dice0:%.03f,dice1:%.03f,dice:%.03f" 
        #             % (epoch, batch_idx, len(dataloaders['tgt_lung_seg_val'].dataset)//inputs.shape[0], 
        #             avg_dice[0], avg_dice[1], np.mean(np.concatenate(all_dices, 0))))

        if ((epoch+1) % opt.eval_cls_times == 0 or epoch+1 == num_epochs):
            # eval infection segmentation and cls
            logger.info("-"*8+"eval infection cls"+"-"*8)
            model.eval()


            val_gt = []
            val_cls_pred = []
            val_seg_pred = []

            for batch_idx, (inputs, labels) in enumerate(dataloaders["tgt_cls_val"], 0):
                inputs = inputs.to(device)
                # adjust label
                val_gt.append(labels.cpu().data.numpy())

                with torch.set_grad_enabled(False):
                    annotation = dataloaders["tgt_cls_val"].dataset.annotations[batch_idx]
                    img_dir = annotation.strip().split(',')[0]
                    img_name = Path(img_dir).name

                    cls_logits, _, seg_logits, _, _ = model(inputs)
                    if opt.do_seg:
                        seg_probs = torch.softmax(seg_logits, dim=1)
                        predicted_mask_onehot = probs2one_hot(seg_probs.detach())

                        # for save
                        predicted_mask = predicted_mask_onehot.squeeze().cpu().numpy() # 3xwxh
                        mask_inone = (np.zeros_like(predicted_mask[0])+predicted_mask[1]*128+predicted_mask[2]*255).astype(np.uint8)

                        # save dir:
                        save_dir = os.path.join(opt.logs, "tgt_cls_val", "ep%03d"%epoch)
                        # 
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                    
                        cv2.imwrite(os.path.join(save_dir, img_name), mask_inone)
                        # seg2cls 
                        preds_cls_seg = (predicted_mask_onehot[:,-1:].sum(-1).sum(-1) > 0).cpu().numpy().astype(np.uint8)
                        val_seg_pred.append(preds_cls_seg)
                

                    # cls
                    #print(cls_logits)
                    if opt.do_cls:
                        probs_cls = torch.softmax(cls_logits, dim=1)
                        preds_cls = (probs_cls[...,1:] > 0.5).type(torch.long)
                        val_cls_pred.append(preds_cls.cpu().data.numpy())

            if not os.path.exists(os.path.join(opt.logs, "cf")):
                os.makedirs(os.path.join(opt.logs, "cf"))

            val_gt = np.concatenate(val_gt, axis=0)


            if opt.do_cls:
                val_cls_pred = np.concatenate(val_cls_pred, axis=0)
                save_cf_png_dir = os.path.join(opt.logs, "cf", "ep%03d_cls_cf.png"%epoch)
                save_metric_dir = os.path.join(opt.logs, "metric_cls.txt")
                result_str = get_results(val_gt, val_cls_pred, save_cf_png_dir, save_metric_dir)            
                logger.info("tgt_cls_val:EP%03d,[cls]: %s" % (epoch, result_str))

            if opt.do_seg:
                val_seg_pred = np.concatenate(val_seg_pred, axis=0)
                # seg2cls
                save_cf_png_dir = os.path.join(opt.logs, "cf", "ep%03d_seg_cf.png"%epoch)
                save_metric_dir = os.path.join(opt.logs, "metric_seg.txt")

                result_str = get_results(val_gt, val_seg_pred, save_cf_png_dir, save_metric_dir)
                logger.info("tgt_seg_val:EP%03d,[seg2cls]: %s" % (epoch, result_str))

    time_elapsed = time.time() - since
    logger.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))




def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./cfgs/experiment.yaml", type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--setgpuid', default=0, type=int)
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(opt, k, v)

    # repalce experiment
    opt.experiment = opt.experiment.replace("only", "mmd")
    
    opt.seg_augment = True
    opt.cls_augment = True

    opt.do_cls_mmd = True
    opt.do_seg = True
    opt.do_cls = True
    opt.do_seg_mmd = False

    opt.eval_cls_times = 50
    opt.eval_times = 50

    opt.random_seed = 1010 * (opt.fold + 1)
    opt.gpuid = opt.setgpuid
    selected_drr_datasets_indexes = np.array(opt.selected_drr_datasets_indexes+opt.selected_drr_datasets_indexes)
    #print(selected_drr_datasets_indexes)

    # # [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    print(selected_drr_datasets_indexes[-1][-1])
    selected_drr_datasets_indexes[2][-1] = 1
    selected_drr_datasets_indexes[3][-1] = 1
    opt.selected_drr_datasets_indexes = [list(_) for _ in list(selected_drr_datasets_indexes)]

    #opt.logs = "logs_experiment03_r5050"

    

    log_dir = "./{}/{}/{}".format("logs", opt.experiment, "fold%d"%opt.fold)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    opt.logs = log_dir
    return opt


if __name__ == "__main__":
    opt = get_argument()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpuid)
    setup_seed(opt.random_seed)


    assert opt.mode == 12, ("opt.mode is not supported in %s" % __file__)
    log_dir = opt.logs

    logger = setup_logger("{}".format(os.path.basename(__file__).split(".")[0]), 
                        save_dir=opt.logs, distributed_rank=0, filename="log.txt")
    logger.info(opt)
    
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
 
    use_pretrained = True
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model_ft = SegClsModule(opt)
    
    train_dataset, tgt_cls_train_dataset, tgt_cls_val_dataset, tgt_lung_seg_val_dataset = genDataset(opt)

    logger.info("-"*8+"train:"+"-"*8)
    logger.info(train_dataset.annotations)

    logger.info("-"*8+"tgt_cls_train:"+"-"*8)
    logger.info(tgt_cls_train_dataset.annotations)

    logger.info("-"*8+"tgt_cls_val:"+"-"*8)
    logger.info(tgt_cls_val_dataset.annotations)

    logger.info("-"*8+"tgt_lung_seg_val:"+"-"*8)
    logger.info(tgt_lung_seg_val_dataset.annotations)

    image_datasets = {'train': train_dataset, 'tgt_cls_train': tgt_cls_train_dataset, 'tgt_cls_val': tgt_cls_val_dataset, "tgt_lung_seg_val": tgt_lung_seg_val_dataset}
    shuffles = {"train": True,'tgt_cls_train': True, 'tgt_cls_val': False, "tgt_lung_seg_val": False}
    batch_sizes_dict = {"train": batch_size,'tgt_cls_train': batch_size, 'tgt_cls_val': 1, "tgt_lung_seg_val": 1}
    drop_lasts = {"train": True,'tgt_cls_train': True, 'tgt_cls_val': False, "tgt_lung_seg_val": False}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes_dict[x], shuffle=shuffles[x], num_workers=4, drop_last=drop_lasts[x]) for x in ['train', 'tgt_cls_train', 'tgt_cls_val', "tgt_lung_seg_val"]}

    # Send the model to GPU
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()

    logger.info("Params to learn:")
    for name,param in model_ft.named_parameters():
        param.requires_grad = True
        logger.info("\t"+name)


  

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=opt.lr)
    criterion = Weighted_Jaccard_loss#nn.CrossEntropyLoss()



    # 学习率decay
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.num_epochs//2 - opt.num_epochs) / float(opt.num_epochs//2 + 1)
        return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lambda_rule)

    ## summarywriter
    writer = SummaryWriter(log_dir=log_dir)

    train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False, log_dir=log_dir, scheduler=scheduler, writer=writer, logger=logger, opt=opt)

