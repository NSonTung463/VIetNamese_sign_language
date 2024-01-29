import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm, notebook
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader,TensorDataset
from collections import defaultdict
import transformers
# from decouple import Config, RepositoryEnv
import random


from data.ds_1 import *
from model.model_1 import *
from utils import *

BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_1")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)


cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_accuracy(model, data):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in torch.utils.data.DataLoader(data, batch_size=len(data)):
            inputs, labels  = inputs.to(cfg.device), labels.to(cfg.device)
            output = model(inputs) # We don't need to run F.softmax
            # loss = criterion(inputs, labels)
            # val_loss += loss.item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        model.train()
        # average_loss = val_loss / len(dataloader)
    return correct / total


from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv('./dataset/train_folded.csv')
# Sử dụng LabelEncoder để chuyển đổi cột 'phrase' thành số
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['phrase'])
tensor_train = torch.load( './data/processed_data/tensor_train_9.pt')
tensor_train = tensor_train[:,:,:1086]
tensor_labels = torch.load( './data/processed_data/tensor_labels_9.pt')
train_dataset = TensorDataset(tensor_train, tensor_labels)

# Set up the dataset and dataloader
# train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
# val_dataset = CustomDataset(val_df, cfg, aug=cfg.train_aug, mode="val")

train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        # num_workers=cfg.num_workers,#//4,
        # pin_memory=cfg.pin_memory,
        # collate_fn=tr_collate_fn,
    )
# val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,#//4,
#         pin_memory=cfg.pin_memory,
#         collate_fn=val_collate_fn,
#     )


input_size = 1086  # Số lượng đặc trưng
hidden_size = 124
output_size = 9

model = SimpleLSTM(cfg.input_size, cfg.hidden_size, cfg.output_size)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()


# Start the training and validation loop
cfg.curr_step = 0
optimizer.zero_grad()
total_grad_norm = None    
total_grad_norm_after_clip = None
i = 0 

if not os.path.exists(f"{cfg.output_dir}/fold{cfg.fold}/"): 
    os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}/")

LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.nwarmup, lr_max=cfg.lr_max,num_training_steps=cfg.epochs, num_cycles=cfg.num_cycles) for step in range(cfg.epochs)]
plot_lr_schedule(LR_SCHEDULE, cfg.epochs)
iters  = []
losses = []
train_acc_list = []
#   val_acc_list = []q
learning_rates=[]
n = 0

for epoch in range(cfg.epochs):
    if cfg.warmup_status == True:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR_SCHEDULE[epoch]
    learning_rates.append(optimizer.param_groups[0]['lr'])
    for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='batch'):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        model.train()
        out = model(inputs)
        loss = criterion(out, labels) # compute the total loss
        loss.backward()   
        
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch
        iters.append(n)
        losses.append(float(loss)/cfg.batch_size) # compute *average* loss
        train_acc_list.append(get_accuracy(model,train_dataset))
        n += 1

    print(f'train_acc: {round(train_acc_list[-1],5)} -  train_loss: {round(losses[-1],3)} -  learning_rate: {round(learning_rates[-1],7)}\n ')

path = f"{cfg.output_dir}/fold{cfg.fold}/"
draw_plot(cfg.epochs,learning_rates,cfg.batch_size,iters,train_acc_list,losses,path)
torch.save({"model": model.state_dict()},path+f"checkpoint_last_seed{cfg.seed}.pth")
print(f"Checkpoint save : " +  path+f"checkpoint_last_seed{cfg.seed}.pth")