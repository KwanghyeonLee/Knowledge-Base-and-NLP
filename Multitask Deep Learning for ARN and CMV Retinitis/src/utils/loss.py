import numpy as np
import torch
import torch.nn as nn

class Loss:
    def __init__(self, opt):
        self.opt = opt
        self.BCE = nn.BCELoss()
        self.CE = nn.CrossEntropyLoss()
    
    
    def __call__(self, output, label):
        if self.opt['model_name'] == 'STL':
            loss = self.STL_loss(output, label)
            loss_list = (loss,)
        elif self.opt['model_name'] == 'ADMTL':
            loss, loss_list = self.Adv_loss(output, label)
        else :
            loss, loss_list = self.MTL_loss(output, label)
        return loss, loss_list
        
    
    def STL_loss(self, output, label):
        output = output.squeeze()
        loss = self.BCE(output.to(torch.float32), label.to(torch.float32))
        return loss
    
    def MTL_loss(self, output, label):
        '''
            output : task loss, loss list
                loss list = (total loss, ARN loss, CMVr loss)
        '''
        out_ARN = output[0]
        out_CMVr = output[1]
        label_ARN = label[0]
        label_CMVr = label[1]
        
        out_ARN = out_ARN.squeeze()
        out_CMVr = out_CMVr.squeeze()
        
        ARN_loss = self.BCE(out_ARN.to(torch.float32), label_ARN.to(torch.float32))
        CMVr_loss = self.BCE(out_CMVr.to(torch.float32), label_CMVr.to(torch.float32))
        
        loss = self.opt['alpha'] * ARN_loss + (1 - self.opt['alpha']) * CMVr_loss
        return loss, (loss, ARN_loss, CMVr_loss)
    
    def Adv_loss(self, output, label):
        '''
            output : total loss, loss list
                loss list = (total loss, task loss, adv loss, diff loss, ARN loss, CMVr loss, diff ARN loss, diff CMVr loss)
                idx {
                    0 : total loss
                    1 : task loss
                    2 : adv loss 
                    3 : diff loss
                    4 : ARN loss
                    5 : CMVr loss
                    6 : diff ARN loss
                    7 : diff CMVr loss
                }
        '''
        out_ARN = output[0]
        out_CMVr = output[1]
        out_adv = output[2]
        (shared, private_ARN, private_CMVr) = output[3]
        
        label_ARN = label[0]
        label_CMVr = label[1]
        label_adv = label[2]
        
        # Task loss
        task_loss, task_loss_list = self.MTL_loss((out_ARN, out_CMVr), (label_ARN, label_CMVr))
        
        # Adversarial loss
        adv_loss = self.CE(out_adv, label_adv)
        
        # Differential loss
        matmul_ARN = torch.matmul(shared.transpose(1, 0), private_ARN)
        matmul_CMVr = torch.matmul(shared.transpose(1, 0), private_CMVr)
        
        diff_ARN_loss = torch.norm(matmul_ARN, p='fro') ** 2
        diff_CMVr_loss = torch.norm(matmul_CMVr, p='fro') ** 2
        diff_loss = diff_ARN_loss + diff_CMVr_loss
        
        # Total loss
        loss = task_loss + self.opt['lambda_adv'] * adv_loss + self.opt['gamma'] * diff_loss
        return loss, (loss, task_loss, adv_loss, diff_loss, task_loss_list[1], task_loss_list[2], diff_ARN_loss, diff_CMVr_loss)