import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.loss import Loss
from src.utils.utils import logger_result
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, opt, model, train_loader, valid_loader, test_loader, **kwargs):
        self.opt = opt
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.criterion = Loss(opt)
    
    def train(self):
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.opt['lr'])
        
        # Criterion (Loss function)
        min_loss = np.inf
        epoch_flag = 0
        
        self.model.train()
        for epoch in range(self.opt['epochs']):
            self.opt['logger'].info(f'Epoch {epoch+1}/{self.opt["epochs"]}')
            
            losses = []
            labels_for_ARN = np.array([])
            preds_for_ARN = np.array([])
            labels_for_CMV = np.array([])
            preds_for_CMV = np.array([])
            
            for i, (feature, (label_ARN, label_CMVr, label_adv)) in enumerate(tqdm(self.train_loader)):
                # to device
                feature = feature.to(self.opt['device'])
                label_ARN = label_ARN.to(self.opt['device'])
                label_CMVr = label_CMVr.to(self.opt['device'])
                label_adv = label_adv.to(self.opt['device'])
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward & Predict
                if self.opt['model_name'] == 'STL':
                    output = self.model(feature)
                    label = label_ARN if self.opt['class_type'] == 1 else label_CMVr
                    out_ARN = output if self.opt['class_type'] == 1 else None
                    out_CMVr = output if self.opt['class_type'] == 2 else None
                elif self.opt['model_name'] == 'ADMTL':
                    out_ARN, out_CMVr, out_adv, emb = self.model(feature)
                    output = (out_ARN, out_CMVr, out_adv, emb)
                    label = (label_ARN, label_CMVr, label_adv)
                else :
                    out_ARN, out_CMVr = self.model(feature)
                    output = (out_ARN, out_CMVr)
                    label = (label_ARN, label_CMVr)
                
                # Loss & Backpropagation
                loss, loss_list = self.criterion(output, label)
                loss.backward()
                optimizer.step()
                
                # Save loss & append label & pred
                losses.append([i.item() for i in loss_list])
                if self.opt['model_name'] != 'STL' or (self.opt['model_name'] == 'STL' and self.opt['class_type'] == 1):
                    labels_for_ARN = np.append(labels_for_ARN, label_ARN.cpu().detach().numpy())
                    preds_for_ARN = np.append(preds_for_ARN, out_ARN.squeeze().cpu().detach().numpy())
                if self.opt['model_name'] != 'STL' or (self.opt['model_name'] == 'STL' and self.opt['class_type'] == 2):
                    labels_for_CMV = np.append(labels_for_CMV, label_CMVr.cpu().detach().numpy())
                    preds_for_CMV = np.append(preds_for_CMV, out_CMVr.squeeze().cpu().detach().numpy())
            
            # Calculate average loss & roc_auc_score
            avg_loss = np.mean(losses, axis=0)
            if labels_for_ARN != []:
                auc_ARN = roc_auc_score(labels_for_ARN, preds_for_ARN)
            if labels_for_CMV != []:
                auc_CMV = roc_auc_score(labels_for_CMV, preds_for_CMV)

            # Get valid loss & roc_auc_score
            valid_avg_loss, valid_auc_score, valid_labels, valid_preds = self.evaluate(mode='valid')
            
            # Save log & Plotter
            if self.opt['model_name'] == 'STL':
                logger_result(self.opt, 'train', avg_loss, auc_ARN if self.opt['class_type'] == 1 else auc_CMV, epoch=epoch)
            else :
                logger_result(self.opt, 'train', avg_loss, (auc_ARN, auc_CMV), epoch=epoch)
            logger_result(self.opt, 'valid', valid_avg_loss, valid_auc_score, epoch=epoch)
            
            # Save best model
            if valid_avg_loss[0] < min_loss:
                min_loss = valid_avg_loss[0]
                torch.save(self.model.state_dict(), self.opt['model_path'])
                epoch_flag = epoch
        self.opt['logger'].info(f'Best model saved at epoch {epoch_flag}')
            
            
            
    def evaluate(self, mode='valid'):
        if mode == 'valid':
            loader = self.valid_loader
        else :
            loader = self.test_loader
        
        self.model.eval()
        losses = []
        labels_for_ARN = np.array([])
        preds_for_ARN = np.array([])
        labels_for_CMV = np.array([])
        preds_for_CMV = np.array([])
        for i, (feature, (label_ARN, label_CMVr, label_adv)) in enumerate(tqdm(loader)):
            # to device
            feature = feature.to(self.opt['device'])
            label_ARN = label_ARN.to(self.opt['device'])
            label_CMVr = label_CMVr.to(self.opt['device'])
            label_adv = label_adv.to(self.opt['device'])
            
            # Forward & Predict
            if self.opt['model_name'] == 'STL':
                output = self.model(feature)
                label = label_ARN if self.opt['class_type'] == 1 else label_CMVr
                out_ARN = output if self.opt['class_type'] == 1 else None
                out_CMVr = output if self.opt['class_type'] == 2 else None
            elif self.opt['model_name'] == 'ADMTL':
                out_ARN, out_CMVr, out_adv, emb = self.model(feature)
                output = (out_ARN, out_CMVr, out_adv, emb)
                label = (label_ARN, label_CMVr, label_adv)
            else :
                out_ARN, out_CMVr = self.model(feature)
                output = (out_ARN, out_CMVr)
                label = (label_ARN, label_CMVr)
            
            # Loss & Backpropagation
            loss, loss_list = self.criterion(output, label)
            
            # Save loss & append label & pred
            losses.append([i.item() for i in loss_list])
            if self.opt['model_name'] != 'STL' or (self.opt['model_name'] == 'STL' and self.opt['class_type'] == 1):
                labels_for_ARN = np.append(labels_for_ARN, label_ARN.cpu().detach().numpy())
                preds_for_ARN = np.append(preds_for_ARN, out_ARN.squeeze().cpu().detach().numpy())
            if self.opt['model_name'] != 'STL' or (self.opt['model_name'] == 'STL' and self.opt['class_type'] == 2):
                labels_for_CMV = np.append(labels_for_CMV, label_CMVr.cpu().detach().numpy())
                preds_for_CMV = np.append(preds_for_CMV, out_CMVr.squeeze().cpu().detach().numpy())
        
        # Calculate average loss & roc_auc_score
        avg_loss = np.mean(losses, axis=0)
        if labels_for_ARN != []:
            auc_ARN = roc_auc_score(labels_for_ARN, preds_for_ARN)
        if labels_for_CMV != []:
            auc_CMV = roc_auc_score(labels_for_CMV, preds_for_CMV)
        
        if self.opt['model_name'] == 'STL':
            if self.opt['class_type'] == 1:
                return avg_loss, auc_ARN, labels_for_ARN, preds_for_ARN
            else:
                return avg_loss, auc_CMV, labels_for_CMV, preds_for_CMV
    
        return avg_loss, (auc_ARN, auc_CMV), (labels_for_ARN, labels_for_CMV), (preds_for_ARN, preds_for_CMV)
    