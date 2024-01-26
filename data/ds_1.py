import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt 

N_TARGET_FRAMES = 124
N_COLS0 = 390
class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def normalize(self,x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
    def forward(self, x):
        
        #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)
        
        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)
        return x
class PreprocessLayer(nn.Module):
    def __init__(self, n_target_frame,n_columns):
        super(PreprocessLayer, self).__init__()
        self.n_target_frame = n_target_frame
        self.n_columns = n_columns
    def forward(self, data, resize=True):
        # Padding with Zeros
        N_FRAMES = data.shape[0] 
        if N_FRAMES < self.n_target_frame:
            zeros_tensor = torch.zeros(self.n_target_frame - N_FRAMES, self.n_columns, dtype=torch.float32)
            data = torch.cat((data, zeros_tensor), dim=0)
        data = data[None]
        tensor_downsampled = F.interpolate(data.unsqueeze(0), size=(self.n_target_frame, self.n_columns), mode='bilinear', align_corners=False)[0]
        data = tensor_downsampled.squeeze(axis=0)
        return data
# augs

def flip(data, flip_array):
    
    data[:,:,0] = -data[:,:,0]
    data = data[:,flip_array]
    return data

def draw_data(data):
    # Vẽ biểu đồ
    print(data.shape)
    plt.imshow(data, interpolation='nearest', origin='upper')
    plt.colorbar()
    plt.show()
def draw_data_3d(data):
    fig = plt.figure()
    print(data.shape)
    
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(data > 0.5)
    ax.scatter(x, y, z)
    plt.show()
    
    

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        
        #input stuff
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        
        symmetry = pd.read_csv(cfg.symmetry_fp).set_index('id')
        flipped_landmarks = symmetry.loc[landmarks]['corresponding_id'].values
        self.flip_array = np.where(landmarks[:,None]==flipped_landmarks[None,:])[1]
        
        
        self.processor = Preprocessing()
        self.preprocesslayer = PreprocessLayer(N_TARGET_FRAMES, N_COLS0 )
        
        #target stuff
        self.flip_aug = cfg.flip_aug

        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        file_id, sequence_id,label = row[['file_id','sequence_id','label']]
        data = self.load_one(file_id, sequence_id)
        seq_len = data.shape[0]
        data = torch.from_numpy(data)
        # print("DATA RAW")
        # draw_data(data)   ## Check
        
        if self.mode == 'train':
            data = self.processor(data)
            # print("After Preprocess")
            # draw_data(data)   ## Check
            # draw_data_3d(data)   ## Check
            
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array) 
                # print("After Flip")
                # draw_data_3d(data)   ## Check
            if self.aug:
                data = self.augment(data)
                # print("After augment")
                # draw_data_3d(data)   ## Check
        
        else: 
            data = self.processor(data)
        data =  self.preprocesslayer(data.reshape(data.shape[0],-1))
        feature_dict = {'input':data,
                        'output': label,
                        }
        return data , label
    
    def augment(self,x):
        x_aug = self.aug(image=x.float())['image']
        return x_aug
    
    def load_one(self, file_id, sequence_id):
        path = self.data_folder + f'{file_id}/{sequence_id}.npy'
        data = np.load(path) # seq_len, 3* nlandmarks
        return data

if __name__ == '__main__':
    pass