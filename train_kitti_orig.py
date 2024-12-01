import os
import sys
import yaml
from SalsaNext import SalsaNext
sys.path.append('/ari/users/ibaskaya/projeler/kittisalsa/utils')

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
from mappings import kitti, kitticolormap,sem2sem, kitti_lm_inv,waymocolormap,waymo,sensor_kitti
from iou_eval import iouEval




if __name__=='__main__':

    num_classes = 20
    batch_number = 24
    batch_size,workers = 8, 16
    max_epochs = 150             
    learning_rate = 0.01     
    warmup_epochs = 1            
    momentum = 0.9              
    lr_decay = 0.99                
    weight_decay = 0.0001        
    batch_size = 8                
    epsilon_w = 0.001 

    frequencies = [0.03150183342534689,
                    0.042607828674502385,
                    0.00016609538710764618,
                    0.00039838616015114444,
                    0.0021649398241338114,
                    0.0018070552978863615,
                    0.0003375832743104974,
                    0.00012711105887399155,
                    3.746106399997359e-05,
                    0.19879647126983288,
                    0.014717169549888214,
                    0.14392298360372,
                    0.0039048553037472045,
                    0.1326861944777486,
                    0.0723592229456223,
                    0.26681502148037506,
                    0.006035012012626033,
                    0.07814222006271769,
                    0.002855498193863172,
                    0.0006155958086189918]

    root = '/ari/users/ibaskaya/projeler/kittisalsa/data/kittiorig'
    train_sequences = [0,1,2,3,4,5,6,7,9,10]
    valid_sequences = [8]
    test_sequences = None
    labels = kitti
    color_map = kitticolormap
    learning_map = sem2sem
    learning_map_inv = kitti_lm_inv
    sensor = sensor_kitti 
    max_points=150000
    gt=True
    transform=False
    iswaymo = False
    istrain = False

    parser = Parser(root,         # directory for data
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
               iswaymo = False)

    
    model = SalsaNext(20,5)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    print(device, ' is used')
    
    
    inverse_frequencies = [1.0 / (f + epsilon_w) for f in frequencies]
    inverse_frequencies[0] = min(inverse_frequencies) / 10
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
            torch.save(model.state_dict(), f'kitti_best_miou_checkpoint.pth')
            print(f"New best mIoU: {best_mean_iou:.4f}. Model saved.")

    
    torch.save(model.cpu().state_dict(), 'waymo_last_checkpoint.pth')

