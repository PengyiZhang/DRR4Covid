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
from sklearn.metrics import roc_auc_score, f1_score

from visual_confuse_matrix import make_confusion_matrix
from dataset import genDataset, genExtraForEvalDataset
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


def get_results(val_labels, val_outs, val_probs, save_cf_png_dir, save_metric_dir):

    # first for probs
    AUC_score = roc_auc_score(val_labels, val_probs)
    F1_score = f1_score(val_labels, val_outs)

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
    result_str = "Sensitivity=%.3f, Specificity=%.3f, PPV=%.3f, NPV=%.3f, FPR=%.3f, FNR=%.3f, FDR=%.3f, ACC=%.3f, AUC=%.3f, F1_score=%.3f\n" % (TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, AUC_score, F1_score)
    
    save_dir = save_metric_dir
    with open(save_dir, "a+") as f:
        f.writelines([result_str])
    return result_str

def eval_model(model, dataloaders, log_dir="./log/", logger=None, opt=None):

    since = time.time()
    if False:#opt.do_seg:
        # eval lung segmentation
        logger.info("-"*8+"eval lung segmentation"+"-"*8)

        model.eval()
        all_dices = []
        all_dices_au = []

        for batch_idx, (inputs, labels) in enumerate(dataloaders["tgt_lung_seg_val"], 0):
            annotation = dataloaders["tgt_lung_seg_val"].dataset.annotations[batch_idx]
            img_dir = annotation.strip().split(',')[0]
            img_name = Path(img_dir).name                
            
            
            inputs = inputs.to(device)
            # adjust labels
            
            labels[labels==opt.xray_mask_value_dict["lung"]] = 1
            
            labels = labels[:,-1].to(device)
            labels = torch.stack([labels == c for c in range(2)], dim=1)

            
            
            with torch.set_grad_enabled(False):
                if opt.use_aux:
                    _, _, seg_logits, _, seg_logits_au = model(inputs)
                else:
                    _, _, seg_logits, _, _ = model(inputs)

                seg_probs = torch.softmax(seg_logits, dim=1)
                predicted_mask = probs2one_hot(seg_probs.detach())

                # change the infection to Lung
                predicted_mask_lung = predicted_mask[:,:-1]
                predicted_mask_lung[:,-1] += predicted_mask[:,-1]
                dices = dice_coef(predicted_mask_lung, labels.detach().type_as(predicted_mask)).cpu().numpy()


                all_dices.append(dices) # [(B,C)]

                predicted_mask_lung = predicted_mask_lung.squeeze().cpu().numpy() # 3xwxh
                mask_inone = (np.zeros_like(predicted_mask_lung[0])+predicted_mask_lung[1]*255).astype(np.uint8)

                # save dir:
                save_dir = os.path.join(opt.logs, "tgt_lung_seg_val", "eval")
                # 
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                cv2.imwrite(os.path.join(save_dir, img_name), mask_inone)

                ###################################################au
                if opt.use_aux:
                    seg_probs_au = torch.softmax(seg_logits_au, dim=1)
                    predicted_mask_au = probs2one_hot(seg_probs_au.detach())

                    # change the infection to Lung
                    predicted_mask_lung_au = predicted_mask_au[:,:-1]
                    predicted_mask_lung_au[:,-1] += predicted_mask_au[:,-1]
                    dices_au = dice_coef(predicted_mask_lung_au, labels.detach().type_as(predicted_mask_au)).cpu().numpy()


                    all_dices_au.append(dices_au) # [(B,C)]

                    predicted_mask_lung_au = predicted_mask_lung_au.squeeze().cpu().numpy() # 3xwxh
                    mask_inone_au = (np.zeros_like(predicted_mask_lung_au[0])+predicted_mask_lung_au[1]*255).astype(np.uint8)

                    # save dir:
                    save_dir_au = os.path.join(opt.logs, "tgt_lung_seg_val_au", "eval")
                    # 
                    if not os.path.exists(save_dir_au):
                        os.makedirs(save_dir_au)
                    
                    cv2.imwrite(os.path.join(save_dir_au, img_name), mask_inone_au)


            avg_dice = np.mean(np.concatenate(all_dices, 0), 0) #


            logger.info("tgt_lung_seg_val:[%d/%d],dice0:%.03f,dice1:%.03f,dice:%.03f" 
                % (batch_idx, len(dataloaders['tgt_lung_seg_val'].dataset)//inputs.shape[0], 
                avg_dice[0], avg_dice[1], np.mean(np.concatenate(all_dices, 0))))
            if opt.use_aux:
                avg_dice_au = np.mean(np.concatenate(all_dices_au, 0), 0) #
                logger.info("tgt_lung_seg_val_au:[%d/%d],dice0:%.03f,dice1:%.03f,dice:%.03f" 
                    % (batch_idx, len(dataloaders['tgt_lung_seg_val'].dataset)//inputs.shape[0], 
                    avg_dice_au[0], avg_dice_au[1], np.mean(np.concatenate(all_dices_au, 0))))

    if True:
        # eval infection segmentation and cls
        logger.info("-"*8+"eval infection cls"+"-"*8)
        model.eval()


        val_gt = []
        val_cls_pred = []
        val_cls_probs = [] # for VOC
        val_seg_pred = [] 
        val_seg_probs = [] # for VOC

        val_seg_probs_au = []
        val_seg_pred_au = [] # for VOC



        for batch_idx, (inputs, labels) in enumerate(dataloaders["tgt_cls_val"], 0):
            inputs = inputs.to(device)
            # adjust label
            val_gt.append(labels.cpu().data.numpy())

            with torch.set_grad_enabled(False):
                annotation = dataloaders["tgt_cls_val"].dataset.annotations[batch_idx]
                img_dir = annotation.strip().split(',')[0]
                img_name = Path(img_dir).name

                if opt.use_aux:
                    cls_logits, _, seg_logits, _, seg_logits_au = model(inputs)
                else:
                    cls_logits, _, seg_logits, _, _ = model(inputs)

                if opt.do_seg:
                    seg_probs = torch.softmax(seg_logits, dim=1)
                    val_seg_probs.append(seg_probs[:,-1:].detach().cpu().view(seg_probs.shape[0], 1, -1).max(-1)[0])

                    predicted_mask_onehot = probs2one_hot(seg_probs.detach())

                    # for save
                    predicted_mask = predicted_mask_onehot.squeeze().cpu().numpy() # 3xwxh
                    mask_inone = (np.zeros_like(predicted_mask[0])+predicted_mask[1]*128+predicted_mask[2]*255).astype(np.uint8)

                    # save dir:
                    save_dir = os.path.join(opt.logs, "tgt_cls_val", "eval")
                    # 
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                
                    cv2.imwrite(os.path.join(save_dir, img_name), mask_inone)
                    # seg2cls 
                    preds_cls_seg = (predicted_mask_onehot[:,-1:].sum(-1).sum(-1) > 0).cpu().numpy().astype(np.uint8)
                    val_seg_pred.append(preds_cls_seg)
                    
                if opt.do_seg and opt.use_aux:
                    seg_probs_au = torch.softmax(seg_logits_au, dim=1)
                    val_seg_probs_au.append(seg_probs_au[:,-1:].detach().cpu().view(seg_probs_au.shape[0], 1, -1).max(-1)[0])

                    predicted_mask_onehot_au = probs2one_hot(seg_probs_au.detach())

                    # for save
                    predicted_mask_au = predicted_mask_onehot_au.squeeze().cpu().numpy() # 3xwxh
                    mask_inone_au = (np.zeros_like(predicted_mask_au[0])+predicted_mask_au[1]*128+predicted_mask_au[2]*255).astype(np.uint8)

                    # save dir:
                    save_dir_au = os.path.join(opt.logs, "tgt_cls_val_au", "eval")
                    # 
                    if not os.path.exists(save_dir_au):
                        os.makedirs(save_dir_au)
                
                    cv2.imwrite(os.path.join(save_dir_au, img_name), mask_inone_au)
                    # seg2cls 
                    preds_cls_seg_au = (predicted_mask_onehot_au[:,-1:].sum(-1).sum(-1) > 0).cpu().numpy().astype(np.uint8)
                    val_seg_pred_au.append(preds_cls_seg_au)                

                # cls
                #print(cls_logits)
                if opt.do_cls:
                    probs_cls = torch.softmax(cls_logits, dim=1)
                    val_cls_probs.append(probs_cls[...,1:].detach().cpu().numpy())
                    preds_cls = (probs_cls[...,1:] > 0.5).type(torch.long)
                    val_cls_pred.append(preds_cls.cpu().data.numpy())

        if not os.path.exists(os.path.join(opt.logs, "cf")):
            os.makedirs(os.path.join(opt.logs, "cf"))

        val_gt = np.concatenate(val_gt, axis=0)


        if opt.do_cls:

            val_cls_pred = np.concatenate(val_cls_pred, axis=0)
            val_cls_probs = np.concatenate(val_cls_probs, axis=0)


            save_cf_png_dir = os.path.join(opt.logs, "cf", "eval_cls_cf.png")
            save_metric_dir = os.path.join(opt.logs, "eval_metric_cls.txt")
            result_str = get_results(val_gt, val_cls_pred, val_cls_probs, save_cf_png_dir, save_metric_dir)            
            logger.info("tgt_cls_val:[cls]: %s" % (result_str))

        if opt.do_seg:
            val_seg_pred = np.concatenate(val_seg_pred, axis=0)
            val_seg_probs = np.concatenate(val_seg_probs, axis=0)

            # seg2cls
            save_cf_png_dir = os.path.join(opt.logs, "cf", "eval_seg_cf.png")
            save_metric_dir = os.path.join(opt.logs, "eval_metric_seg.txt")

            result_str = get_results(val_gt, val_seg_pred, val_seg_probs, save_cf_png_dir, save_metric_dir)
            logger.info("tgt_seg_val:[seg2cls]: %s" % (result_str))

        if opt.do_seg and opt.use_aux:
            val_seg_pred_au = np.concatenate(val_seg_pred_au, axis=0)
            val_seg_probs_au = np.concatenate(val_seg_probs_au, axis=0)

            # seg2cls
            save_cf_png_dir_au = os.path.join(opt.logs, "cf", "eval_seg_au_cf.png")
            save_metric_dir_au = os.path.join(opt.logs, "eval_metric_seg_au.txt")

            result_str_au = get_results(val_gt, val_seg_pred_au, val_seg_probs_au, save_cf_png_dir_au, save_metric_dir_au)
            logger.info("tgt_seg_au_val:[seg2cls]: %s" % (result_str_au))

    time_elapsed = time.time() - since
    logger.info("Eval complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

def extra_eval_model(model, dataloaders, log_dir="./log/", logger=None, opt=None):
    since = time.time()
    if True:
        # eval infection segmentation and cls
        logger.info("-"*8+"extra eval infection cls"+"-"*8)
        model.eval()


        val_gt = []
        val_cls_pred = []
        val_cls_probs = [] # for VOC
        val_seg_pred = [] 
        val_seg_probs = [] # for VOC

        val_seg_probs_au = []
        val_seg_pred_au = [] # for VOC


        annotations = dataloaders["tgt_cls_extra_val"].dataset.annotations
        for batch_idx, (inputs, labels) in enumerate(dataloaders["tgt_cls_extra_val"], 0):
            inputs = inputs.to(device)
            # adjust label
            val_gt.append(labels.cpu().data.numpy())

            with torch.set_grad_enabled(False):
                annotation = annotations[batch_idx]
                img_dir = annotation.strip().split(',')[0]
                img_name = Path(img_dir).name
                print(batch_idx, len(annotations))

                if opt.use_aux:
                    cls_logits, _, seg_logits, _, seg_logits_au = model(inputs)
                else:
                    cls_logits, _, seg_logits, _, _ = model(inputs)

                if opt.do_seg:
                    seg_probs = torch.softmax(seg_logits, dim=1)
                    val_seg_probs.append(seg_probs[:,-1:].detach().cpu().view(seg_probs.shape[0], 1, -1).max(-1)[0])

                    predicted_mask_onehot = probs2one_hot(seg_probs.detach())

                    # for save
                    predicted_mask = predicted_mask_onehot.squeeze().cpu().numpy() # 3xwxh
                    mask_inone = (np.zeros_like(predicted_mask[0])+predicted_mask[1]*128+predicted_mask[2]*255).astype(np.uint8)

                    # save dir:
                    save_dir = os.path.join(opt.logs, "tgt_cls_extra_val", "eval")
                    # 
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                
                    cv2.imwrite(os.path.join(save_dir, img_name), mask_inone)
                    # seg2cls 
                    preds_cls_seg = (predicted_mask_onehot[:,-1:].sum(-1).sum(-1) > 0).cpu().numpy().astype(np.uint8)
                    val_seg_pred.append(preds_cls_seg)
                    
                if opt.do_seg and opt.use_aux:
                    seg_probs_au = torch.softmax(seg_logits_au, dim=1)
                    val_seg_probs_au.append(seg_probs_au[:,-1:].detach().cpu().view(seg_probs_au.shape[0], 1, -1).max(-1)[0])

                    predicted_mask_onehot_au = probs2one_hot(seg_probs_au.detach())

                    # for save
                    predicted_mask_au = predicted_mask_onehot_au.squeeze().cpu().numpy() # 3xwxh
                    mask_inone_au = (np.zeros_like(predicted_mask_au[0])+predicted_mask_au[1]*128+predicted_mask_au[2]*255).astype(np.uint8)

                    # save dir:
                    save_dir_au = os.path.join(opt.logs, "tgt_cls_extra_val_au", "eval")
                    # 
                    if not os.path.exists(save_dir_au):
                        os.makedirs(save_dir_au)
                
                    cv2.imwrite(os.path.join(save_dir_au, img_name), mask_inone_au)
                    # seg2cls 
                    preds_cls_seg_au = (predicted_mask_onehot_au[:,-1:].sum(-1).sum(-1) > 0).cpu().numpy().astype(np.uint8)
                    val_seg_pred_au.append(preds_cls_seg_au)                

                # cls
                #print(cls_logits)
                if opt.do_cls:
                    probs_cls = torch.softmax(cls_logits, dim=1)
                    val_cls_probs.append(probs_cls[...,1:].detach().cpu().numpy())
                    preds_cls = (probs_cls[...,1:] > 0.5).type(torch.long)
                    val_cls_pred.append(preds_cls.cpu().data.numpy())

        if not os.path.exists(os.path.join(opt.logs, "cf")):
            os.makedirs(os.path.join(opt.logs, "cf"))

        val_gt = np.concatenate(val_gt, axis=0)


        if opt.do_cls:

            val_cls_pred = np.concatenate(val_cls_pred, axis=0)
            val_cls_probs = np.concatenate(val_cls_probs, axis=0)


            save_cf_png_dir = os.path.join(opt.logs, "cf", "extra_eval_cls_cf.png")
            save_metric_dir = os.path.join(opt.logs, "extra_eval_metric_cls.txt")
            result_str = get_results(val_gt, val_cls_pred, val_cls_probs, save_cf_png_dir, save_metric_dir)            
            logger.info("tgt_cls_extra_val:[cls]: %s" % (result_str))

        if opt.do_seg:
            val_seg_pred = np.concatenate(val_seg_pred, axis=0)
            val_seg_probs = np.concatenate(val_seg_probs, axis=0)

            # seg2cls
            save_cf_png_dir = os.path.join(opt.logs, "cf", "extra_eval_seg_cf.png")
            save_metric_dir = os.path.join(opt.logs, "extra_eval_metric_seg.txt")

            result_str = get_results(val_gt, val_seg_pred, val_seg_probs, save_cf_png_dir, save_metric_dir)
            logger.info("tgt_seg_extra_val:[seg2cls]: %s" % (result_str))

        if opt.do_seg and opt.use_aux:
            val_seg_pred_au = np.concatenate(val_seg_pred_au, axis=0)
            val_seg_probs_au = np.concatenate(val_seg_probs_au, axis=0)

            # seg2cls
            save_cf_png_dir_au = os.path.join(opt.logs, "cf", "extra_eval_seg_au_cf.png")
            save_metric_dir_au = os.path.join(opt.logs, "extra_eval_metric_seg_au.txt")

            result_str_au = get_results(val_gt, val_seg_pred_au, val_seg_probs_au, save_cf_png_dir_au, save_metric_dir_au)
            logger.info("tgt_seg_au_extra_val:[seg2cls]: %s" % (result_str_au))

    time_elapsed = time.time() - since
    logger.info("Extra_Eval complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))



def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./cfgs/experiment.yaml", type=str)
    #parser.add_argument('--setseed', default=2020, type=int)
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

    # opt.random_seed = opt.setseed
    opt.random_seed = 1010 * (opt.fold + 1)
    opt.gpuid = opt.setgpuid

    selected_drr_datasets_indexes = np.array(opt.selected_drr_datasets_indexes+opt.selected_drr_datasets_indexes)
    #print(selected_drr_datasets_indexes)

    # # [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    print(selected_drr_datasets_indexes[-1][-1])
    selected_drr_datasets_indexes[2][-1] = 1
    selected_drr_datasets_indexes[3][-1] = 1
    opt.selected_drr_datasets_indexes = [list(_) for _ in list(selected_drr_datasets_indexes)]

    log_dir = "./{}/{}/{}".format("logs_bk", opt.experiment, "fold%d"%opt.fold)
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
                        save_dir=opt.logs, distributed_rank=0, filename="log_eval.txt")
    logger.info(opt)
    
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
 
    use_pretrained = True
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model_ft = SegClsModule(opt)
    
    train_dataset, tgt_cls_train_dataset, tgt_cls_val_dataset, tgt_lung_seg_val_dataset = genDataset(opt)
    tgt_cls_extra_val_dataset = genExtraForEvalDataset(opt)

    logger.info("-"*8+"train:"+"-"*8)
    logger.info(train_dataset.annotations)

    logger.info("-"*8+"tgt_cls_train:"+"-"*8)
    logger.info(tgt_cls_train_dataset.annotations)

    logger.info("-"*8+"tgt_cls_val:"+"-"*8)
    logger.info(tgt_cls_val_dataset.annotations)

    logger.info("-"*8+"tgt_cls_extra_val:"+"-"*8)
    logger.info(tgt_cls_extra_val_dataset.annotations)

    # logger.info("-"*8+"tgt_lung_seg_val:"+"-"*8)
    # logger.info(tgt_lung_seg_val_dataset.annotations)




    image_datasets = {'train': train_dataset, 'tgt_cls_train': tgt_cls_train_dataset, 'tgt_cls_val': tgt_cls_val_dataset, 'tgt_cls_extra_val': tgt_cls_extra_val_dataset, "tgt_lung_seg_val": tgt_lung_seg_val_dataset}
    shuffles = {"train": True,'tgt_cls_train': True, 'tgt_cls_val': False, 'tgt_cls_extra_val': False, "tgt_lung_seg_val": False}
    batch_sizes_dict = {"train": batch_size,'tgt_cls_train': batch_size, 'tgt_cls_val': 1, 'tgt_cls_extra_val': 1, "tgt_lung_seg_val": 1}
    drop_lasts = {"train": True,'tgt_cls_train': True, 'tgt_cls_val': False, 'tgt_cls_extra_val': False, "tgt_lung_seg_val": False}
    number_worker_dict = {"train": 4,'tgt_cls_train': 4, 'tgt_cls_val': 0, 'tgt_cls_extra_val': 0, "tgt_lung_seg_val": 0}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes_dict[x], shuffle=shuffles[x], num_workers=number_worker_dict[x], drop_last=drop_lasts[x]) for x in ['train', 'tgt_cls_train', 'tgt_cls_val', 'tgt_cls_extra_val', "tgt_lung_seg_val"]}

    # Send the model to GPU
    
    weight_path = os.path.join(log_dir, "latest.pth")
    model_ft.load_state_dict(torch.load(weight_path))
    model_ft = model_ft.to(device)
    model_ft.eval()

    #eval_model(model_ft, dataloaders_dict, log_dir=log_dir, logger=logger, opt=opt)
    extra_eval_model(model_ft, dataloaders_dict, log_dir=log_dir, logger=logger, opt=opt)

