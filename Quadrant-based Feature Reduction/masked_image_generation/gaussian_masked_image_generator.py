#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:21:40 2023

@author: mikewang
"""
from mask_generate import mask_generate
from gaussian_mask_generate import gaussian_mask_generate
from itertools import chain, combinations
import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
#%% define how to generate image quaderents 
k = 1
num_parts = 4
# grey_area = [0]  ###block number
image = cv2.imread('1f8f08ea-b5b3-4f68-94d4-3cc071b7dce8.png')

def generate_combinations(num_parts):
    list_parts = list(range(num_parts))
    combination_list = []
    for i in range(num_parts+1):
        combination_list_curr = list(combinations(list_parts,i))
        combination_list = combination_list + combination_list_curr
    return combination_list

def generate_masked_images(I,k, num_parts, save_dir, isSmall, mask_type, std=None):
    grey_area_list_tuple = generate_combinations(num_parts)
    grey_area_list = []
    for i in range(len(grey_area_list_tuple)):
        grey_area_list.append(list(grey_area_list_tuple[i]))
    # print(grey_area_list)
    for i in range(len(grey_area_list)):
        print(list(grey_area_list[i]))
        if mask_type == 0:
            I_m = mask_generate(I, k, grey_area_list[i])
        elif mask_type == 1:
            I_m = gaussian_mask_generate(I,k,grey_area_list[i], ksize=(599, 799), std=std)
        # plt.imshow(I_m)
        print(I_m.shape)
        if isSmall == False: 
            if len(grey_area_list[i]) > 0:
                for j in range(len(grey_area_list[i])):
                    grey_area_list[i][j] = grey_area_list[i][j] + 1
                myString = ''.join(map(str,grey_area_list[i]))
                # print("/output_"+myString+".png")
                I_m.save(save_dir+"/output_"+myString+".png")
            else: 
                I_m.save(save_dir+"/output_.png")
        else: 
            newsize = (256, 256)
            I_m_copy = I_m.copy()
            # plt.imshow(I_m_copy)
            I_m_r = cv2.resize(I_m_copy,newsize)
            I_m_r_image = Image.fromarray(I_m_r, "RGB")
            if len(grey_area_list[i]) > 0:
                for j in range(len(grey_area_list[i])):
                    grey_area_list[i][j] = grey_area_list[i][j] + 1
                myString = ''.join(map(str,grey_area_list[i]))
                # print("/output_"+myString+".png")
                I_m_r_image.save(save_dir+"/output_"+myString+".png")
            else: 
                I_m_r_image.save(save_dir+"/output_.png")
    

#%%
# --------------------------- mask as pure black ----------------------------
save_dir = "output_shap/Guassian_std_100_small"
# save_dir = "output_shap/black_masked_small"
generate_masked_images(image, k, num_parts, save_dir, True, 1, std=100)
#%%
temp = generate_combinations(4)