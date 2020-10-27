import itk
import numpy as np
import time
import os
import sys
import glob
import json
import csv
import pandas as pd
import cv2

from pathlib import Path

import ProjectorsModule as pm
import ReadWriteImageModule as rw


import random 

import math 

# 01: 1, 02: 0.5
focal_lenght = 1000
# Define projector for generation of DRR from 3D model (Digitally Reconstructed Radiographs)
projector_info = {'Name': 'SiddonGpu', 
                  'threadsPerBlock_x': 16,
                  'threadsPerBlock_y': 16,
                  'threadsPerBlock_z': 1,
                  'focal_lenght': focal_lenght,
                  'DRRspacing_x': 0.2756, # 0.5, 1
                  'DRRspacing_y': 0.2756,
                  'DRR_ppx': 3.6180,
                  'DRR_ppy': 3.6180,
                  'DRRsize_x': 1024,
                  'DRRsize_y': 1024,
                  }



def initialize_projector(Projector_info, model_filepath="./coronacases_001.nii.gz", maskFileName=None, ClassWeights=[1.0,1.0,1.0]):
    Projector = pm.projector_factory(Projector_info, model_filepath, maskFileName,ClassWeights=ClassWeights)
    return Projector

def gen_drr(Projector, transform_parameters, save_dir, stem, number):
    DRR, Mask_infection, Mask_lung, Mask_value = Projector.compute(transform_parameters) # rotX, rotY, rotZ, transX, transY, transZ
    # Save DRR
    idm_dir = "DRR"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(DRR, itk.Image[itk.F,2], save_path_dir, extension = '.tif')

    # Mask infection
    idm_dir = "Infection"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_infection, itk.Image[itk.F,2], save_path_dir, extension = '.tif')


    # Lung infection
    idm_dir = "Lung"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_lung, itk.Image[itk.F,2], save_path_dir, extension = '.tif') 

    idm_dir = "Value"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_value, itk.Image[itk.F,2], save_path_dir, extension = '.tif')   

    # -- or Rescale write as 8 bit .png
    # Initialize types
    idm_dir = "img"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d.png"%number)


    OutputPngImageType = itk.Image[itk.UC,2]
    FixedImageType = itk.Image[itk.F,2]

    RescaleFilterType = itk.RescaleIntensityImageFilter[FixedImageType, OutputPngImageType]
    rescaler = RescaleFilterType.New()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.SetInput(DRR)
    drr_final = rescaler.GetOutput()
    newDRRArray = itk.GetArrayViewFromImage(drr_final)
    newDRRArray = cv2.rotate(newDRRArray, rotateCode = cv2.ROTATE_180)
    newDRRArray = cv2.flip(newDRRArray, 1)

    cv2.imwrite(save_path_dir, newDRRArray)

def gen_drr_v2(Projector, transform_parameters, save_dir, stem, number):
    DRR, Mask_infection, Mask_lung, Mask_value = Projector.compute(transform_parameters) # rotX, rotY, rotZ, transX, transY, transZ
    # Save DRR
    # idm_dir = "DRR"
    # if not os.path.exists(os.path.join(save_dir, idm_dir)):
    #     os.makedirs(os.path.join(save_dir, idm_dir))
    # save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    # rw.ImageWriter(DRR, itk.Image[itk.F,2], save_path_dir, extension = '.tif')

    # Mask infection
    idm_dir = "Infection"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_infection, itk.Image[itk.F,2], save_path_dir, extension = '.tif')


    # Lung infection
    idm_dir = "Lung"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_lung, itk.Image[itk.F,2], save_path_dir, extension = '.tif') 

    idm_dir = "Value"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d"%number)

    rw.ImageWriter(Mask_value, itk.Image[itk.F,2], save_path_dir, extension = '.tif')   

    # -- or Rescale write as 8 bit .png
    # Initialize types
    idm_dir = "img"
    if not os.path.exists(os.path.join(save_dir, idm_dir)):
        os.makedirs(os.path.join(save_dir, idm_dir))
    save_path_dir = os.path.join(save_dir, idm_dir, stem+"%03d.png"%number)


    OutputPngImageType = itk.Image[itk.UC,2]
    FixedImageType = itk.Image[itk.F,2]

    RescaleFilterType = itk.RescaleIntensityImageFilter[FixedImageType, OutputPngImageType]
    rescaler = RescaleFilterType.New()
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.SetInput(DRR)
    drr_final = rescaler.GetOutput()
    newDRRArray = itk.GetArrayViewFromImage(drr_final)
    newDRRArray = cv2.rotate(newDRRArray, rotateCode = cv2.ROTATE_180)
    newDRRArray = cv2.flip(newDRRArray, 1)

    cv2.imwrite(save_path_dir, newDRRArray)


def get_random_transform():
    transform_parameters = [0,0,0,0,0,0]
    index = random.randint(0,10000) % 6
    if index == 0:
        high_alpha_x, low_alpha_x = 45, -45
        alpha_x = random.randint(low_alpha_x, high_alpha_x) / 180.0 * 3.14
        transform_parameters = [alpha_x, 0, 0, 0, 0, 0]
    elif index == 1:
        high_alpha_z, low_alpha_z = 45, -45
        alpha_z = random.randint(low_alpha_z, high_alpha_z) / 180.0 * 3.14
        transform_parameters = [0, 0, alpha_z, 0, 0, 0]        
    elif index == 2:
        high_alpha_y, low_alpha_y = 30, 0
        alpha_y = random.randint(low_alpha_y, high_alpha_y) / 180.0 * 3.14
        transform_parameters = [0, alpha_y, 0, 0, 0, 0]
    elif index == 3: 
        transform_parameters = [0, 0, 0, random.randint(-100, 100), 0, 0]
    elif index == 4: 
        transform_parameters = [0, 0, 0, 0, random.randint(0, 100), 0]    
    else: 
        transform_parameters = [0, 0, 0, 0, 0, random.randint(-100, 100)]

    print(transform_parameters)

    return transform_parameters


def gen_drr_img():

    random.seed(2020)

    ClassWeights = [24.0, 24.0, 1.0]
    ClassWeights[0] = ClassWeights[1]
    print(ClassWeights)
    save_dir = "./drr_%d_%.1f_%.1f" % (focal_lenght, ClassWeights[1], ClassWeights[2])


    # model_filepaths = ["../../data/volume/IMG/coronacases_%03d.nii.gz" % (i+1) for i in range(10)]
    # model_maskpaths = ["../../data/volume/GT/coronacases_%03d.nii.gz" % (i+1) for i in range(10)]
    root_dirs = "../../data/Task06_Lung" # 
    mask_name = "labelsTr"
    img_name = "imagesTr"

    model_filepaths = [os.path.join(root_dirs, img_name, _) for _ in sorted(os.listdir(os.path.join(root_dirs, img_name)))]
    model_maskpaths = [os.path.join(root_dirs, mask_name, _) for _ in sorted(os.listdir(os.path.join(root_dirs, mask_name)))]

    assert len(model_filepaths) == len(model_maskpaths)

    num = 20
    file_indexes = [11,24,40,54]
    number = file_index * num
    
    for idx, (model_filepath, model_maskpath) in enumerate(zip(model_filepaths[file_index:], model_maskpaths[file_index:]), file_index):
        model_Path = Path(model_filepath)
        Projector = initialize_projector(projector_info, model_filepath, model_maskpath, ClassWeights=ClassWeights)

        for i in range(num):
            print(model_Path, i)
            transform_parameters = get_random_transform()
            gen_drr(Projector, transform_parameters, save_dir, model_Path.stem, number)
            number = number + 1
        del Projector
    


if __name__ == "__main__":    
    gen_drr_img()
