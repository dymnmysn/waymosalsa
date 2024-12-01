import os
import sys
import yaml
from SalsaNext import SalsaNext
sys.path.append('/ari/users/ibaskaya/projeler/waymosalsa/utils')

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from fastfill import FastFill
from scale3d import RandomRescaleRangeImage
from dskittiwaymo import SegmentationDataset

from metric_miou import calculate_classwise_intersection_union,calculate_final_miou_from_batches, calculate_miou
from printiou import print_miou_kitti as print_miou_results
from lovasz import Lovasz_softmax
from parser import Parser
from mappings import kitti, kitticolormap,sem2sem, kitti_lm_inv,waymocolormap,waymo,sensor_kitti,sensor_waymo,waymo_lmap, waymo_lmap_inv,waymovalidfreqs
from iou_eval import iouEval


if __name__=='__main__':

    num_classes = 23
    batch_number = 24
    batch_size,workers = 24, 24
    max_epochs = 150             
    learning_rate = 0.01     
    warmup_epochs = 1            
    momentum = 0.9              
    lr_decay = 0.99                
    weight_decay = 0.0001        
    batch_size = 8                
    epsilon_w = 0.001 

    freqsum = sum(waymovalidfreqs)
    frequencies = [i/freqsum for i in waymovalidfreqs]

    root = '/ari/users/ibaskaya/projeler/waymosalsa/data/waymo'
    
    train_sequences = [0,1,2,3,4,5,6,7,9,10]
    valid_sequences = [8]
    test_sequences = None
    labels = waymo
    color_map = waymocolormap
    learning_map = waymo_lmap
    learning_map_inv = waymo_lmap_inv
    sensor = sensor_waymo
    max_points=170000
    gt=True
    transform=False
    iswaymo = False
    istrain = False

    model = SalsaNext(23,5)

    #Bura checkpoint
    checkpoint = '/ari/users/ibaskaya/projeler/waymosalsa/waymo_best_model_state_dict.pth'
    model.load_state_dict(torch.load(checkpoint)) 
    learning_rate = 0.006426

    parser = Parser(root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,
               iswaymo = True,
               concat=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    print(device, ' is used')
    
    
    inverse_frequencies = [1.0 / (f + epsilon_w) for f in frequencies]
    inverse_frequencies[0] = min(inverse_frequencies) / 50
    criterion_nll = nn.NLLLoss(weight=torch.tensor(inverse_frequencies).to(device))
    criterion_lovasz = Lovasz_softmax(ignore=0, from_logits=False)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    def warmup_lr_scheduler(optimizer, warmup_epochs, initial_lr):
        def lr_lambda(epoch):
            return epoch / warmup_epochs if epoch < warmup_epochs else 1
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, learning_rate)

    best_mean_iou = 0.0
    metric = iouEval(num_classes,device,0)

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        for i, batch in tqdm(enumerate(parser.trainloader), total=len(parser.trainloader)):
            images, masks = batch[0],batch[2]
            images, masks = images.to(torch.float32).to(device), masks.to(torch.long).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion_nll(torch.log(outputs), masks) + criterion_lovasz(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        print(f"Epoch [{epoch+1}/{max_epochs}], Training Loss: {running_loss / len(parser.trainloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation phase
        model.eval()
        miou_total = 0.0

        metric.reset()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(parser.validloader)):
                images, masks = batch[0],batch[2]
                images, masks = images.to(torch.float32).to(device), masks.to(torch.long).to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                metric.addBatch(preds,masks)


        # Calculate and display mIoU metrics
        mean_iou = metric.getIoU()[0].item()
        accval = metric.getacc().item()

        print(f"Epoch [{epoch+1}/{max_epochs}], Validation mIoU: {mean_iou:.4f}")
        print(f"Epoch [{epoch+1}/{max_epochs}], Validation Acc.: {accval:.4f}")
        print('################################################')

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), f'waymo_best_miou_checkpoint.pth')
            print(f"New best mIoU: {best_mean_iou:.4f}. Model saved.")

    
    torch.save(model.cpu().state_dict(), 'waymo_last_checkpoint.pth')

