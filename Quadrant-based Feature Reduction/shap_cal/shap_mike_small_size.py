#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:32:10 2022

@author: mikewang
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import hshap
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from itertools import combinations 
import math
# import cv2
# select device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
M = 4 # total number of players
#%%

model_directory = 'prediction_model/'
filename = model_directory + 'model.pt'
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(filename, map_location=device))
model = model.to(device)
model.eval()
torch.cuda.empty_cache()
model = nn.Sequential(
    model,
    nn.Softmax(dim=1),
)
print(model)


#%%
image_path ='/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.580.697/Interpretable-ML/output_shap/1f8f08ea-b5b3-4f68-94d4-3cc071b7dce8.png'
# resized_image_path ='/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.580.697/Interpretable-ML/output_shap/original.png'
image_o = Image.open(image_path)
newsize = (256, 256)
image = image_o.resize(newsize)
# image.save(resized_image_path)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)


print("before resize")
image_to = transform(image_o).to(device)
prep_img_o = torch.unsqueeze(image_to, 0)
prob_scores_o = model(prep_img_o).detach().numpy()[0]
print(prob_scores_o)
cl_o, baseline_v_o = np.argmax(prob_scores_o), prob_scores_o[np.argmax(prob_scores_o)]

print("after resize")
image_t = transform(image).to(device)
prep_img = torch.unsqueeze(image_t, 0)
prob_scores = model(prep_img).detach().numpy()[0]
print(prob_scores)
# cl, baseline_v = np.argmax(prob_scores), prob_scores[np.argmax(prob_scores)]
cl = 1





#%%
directory = '/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.580.697/Interpretable-ML/output_shap/small_size'
# mean = torch.tensor([0.485, 0.456, 0.406])
# std = torch.tensor([0.229, 0.224, 0.225])
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
# )

#%%
def exclude_k_matrics_calculation(k,directory, M):
    exclude_player2_list = []
    exclude_player2_team_index_list = []
    player_list = []
    probList = []
    contains_player2_list = []
    contains_player2_team_index_list = []
    team_index = 0 
    all_players = list(range(1,M+1))
    
    for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f) and f[-4:] == '.png' and "output" in filename:
                ext = filename[7:-4]
                # print(ext)
                players_excluded = list(ext)
                # print(players_excluded)
                players_excluded = [int(i) for i in players_excluded]
                players = all_players.copy()
                for excludedPalyerId in players_excluded: 
                    players.remove(excludedPalyerId)
                if k in players:
                    # print("--- Ture")
                    contains_player2_list.append(players)
                    contains_player2_team_index_list.append(team_index)
                else: 
                    # print("--- False")
                    exclude_player2_list.append(players)
                    exclude_player2_team_index_list.append(team_index)
                # print(filename)
                # print(players)
                player_list.append(players)
                image = Image.open(directory+"/"+filename)
                image_t = transform(image).to(device)
                prep_img = torch.unsqueeze(image_t, 0)
                
                # print(prep_img.shape)
                prob_scores = model(prep_img).detach().numpy()[0]
                # print(prob_scores)
                p = prob_scores[cl]
                probList.append(p)
                # print(p)
                team_index += 1
    return exclude_player2_list, exclude_player2_team_index_list, player_list, probList, contains_player2_list, contains_player2_team_index_list

#%% calcualte shaply
def shap_k_calulation(k, exclude_playerk_list, exclude_playerk_team_index_list, player_list, probList, contains_playerk_list, contains_playerk_team_index_list):
    shap_k = 0
    for i, teams_withk in enumerate(contains_playerk_list):
        
        teams_withoutk = teams_withk.copy()
        teams_withoutk.remove(k)
        subset_size = len(teams_withoutk)
        prob_excluded = probList[exclude_playerk_team_index_list[exclude_playerk_list.index(teams_withoutk)]]
        print("---")
        print(teams_withoutk)
        print(prob_excluded)
        # print(exclude_playerk_team_index_list[exclude_playerk_list.index(teams_withoutk)])
        prob_with = probList[contains_playerk_team_index_list[i]]
        print(teams_withk)
        print(prob_with)
        shap_k = shap_k + (math.factorial(subset_size)*math.factorial(M-subset_size-1))*(prob_with-prob_excluded)/math.factorial(M)
        # print(contains_playerk_team_index_list[i])
    print("the shap value of {} is: {} ".format(k,shap_k))
    return shap_k


def shap_main(k_list, directory, M):
    shap_value_list = []
    for k in k_list:     
        print("====================== looking at k=" + str(k))
        exclude_playerk_list, exclude_playerk_team_index_list, player_list, probList, contains_playerk_list, contains_playerk_team_index_list = exclude_k_matrics_calculation(k,directory, M)
        # print(probList)
        shap_k = shap_k_calulation(k, exclude_playerk_list, exclude_playerk_team_index_list, player_list, probList, contains_playerk_list, contains_playerk_team_index_list)
        shap_value_list.append([k,shap_k])
    return shap_value_list


#%%
k_list = list(range(1,M+1))
shap_value_list = shap_main(k_list,directory, M)
#%%
image_out = np.asarray(image_o).copy()

red_ratio_array = np.array([[0.03864981288400789, 0.1004361535112063],[0.7422940880060196, 0.04874143462317686]])
for i in range(2):
    for j in range(2):
        image_out[600*i:600*(i+1),800*j:800*(j+1),0] = image_out[600*i:600*(i+1),800*j:800*(j+1),0]+255*red_ratio_array[int(i),int(j)]

plt.imshow(image_out)
plt.show()



#%% load image

#%% generate combination index


#%% mask generation method



