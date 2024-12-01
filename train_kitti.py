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



if __name__=='__main__':

    num_classes = 20
    batch_number = 24
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

    
    model = SalsaNext(20,5)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    print(device, ' is used')

    ff = FastFill(tofill=0, indices=[0,1,2,3,4])

    transform_train = A.Compose([
        #A.Resize(height=64, width=2048, interpolation=cv2.INTER_NEAREST, p=1), 
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.0, rotate_limit=0, 
                        border_mode=cv2.BORDER_WRAP, interpolation=cv2.INTER_NEAREST,
                        p=0.5),  
        A.RandomCrop(height = 64, width = 2048, p=1),
        #A.PadIfNeeded(min_height=64, min_width=2048, border_mode=0, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),  
        #A.CoarseDropout(max_holes=2, max_height=64, max_width=256, min_holes=1, min_height=1, min_width=1, fill_value=0, p=1), 
        ToTensorV2() 
    ], additional_targets={'mask': 'image'})
    transform_valid = A.Compose([
        A.Resize(height=64, width=2048, interpolation=cv2.INTER_NEAREST, p=1),  
        ToTensorV2()  
    ], additional_targets={'mask': 'image'})

    pretransform = None

    train_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/kittisalsa/data/kitti', 
                                    split = 'training', transform=transform_train, 
                                    pretransform=pretransform, fastfill=ff, iswaymo=False, width=2048)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_number, shuffle=True, drop_last=True, num_workers=10)

    validation_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/kittisalsa/data/kitti', 
                                        split = 'validation', transform=transform_valid, 
                                        pretransform=None, fastfill=ff, iswaymo=False, width=2048)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_number, shuffle=False, drop_last=True, num_workers=10)

    
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
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        for i, (images, masks) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            images, masks = images.to(torch.float32).to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion_nll(torch.log(outputs), masks) + criterion_lovasz(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{max_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation phase
        model.eval()
        miou_total = 0.0
        batch_results = []

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(validation_dataloader)):
                images, masks = images.to(torch.float32).to(device), masks.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)

                miou = calculate_miou(preds, masks, num_classes, ignore_index=0)
                cwiou = calculate_classwise_intersection_union(preds, masks)
                batch_results.append(cwiou)

                miou_total += miou

        # Calculate and display mIoU metrics
        classwise_iou, mean_iou, total_iou = calculate_final_miou_from_batches(batch_results)
        print_miou_results(classwise_iou, mean_iou, total_iou)
        avg_miou = miou_total / len(validation_dataloader)
        print(f"Epoch [{epoch+1}/{max_epochs}], Validation mIoU: {avg_miou:.4f}")
        print('################################################')

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), f'kitti_best_model_state_dict.pth')
            print(f"New best mIoU: {best_mean_iou:.4f}. Model saved.")

    
    torch.save(model.cpu().state_dict(), 'kitti_last_model_state_dict.pth')

