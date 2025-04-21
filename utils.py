from ctypes import Union
import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision import transforms
import warnings
import torch.utils.data
from peft import LoraConfig, get_peft_model, PeftModelForSequenceClassification
# from transformers import EsmConfig, EsmTokenizer, EsmForMaskedLM, EsmForSequenceClassification
from Bio.PDB import PDBParser, PPBuilder
import sys
from logger import *
from time import gmtime, strftime
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import *
import random


def accuracy(pred, target):
    try:
        acc = (pred.argmax(dim=-1) == target).float().mean()
    except:
        acc = 0
    return acc

def f1_max(pred, target):
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()

def calculate_accuracy(pred, target):
    """
    计算预测准确率。
    参数:
        pred (torch.Tensor): 模型输出的预测概率。
        target (torch.Tensor): 真实的目标标签。
    返回:
        accuracy (float): 预测的准确率。
    """
    # 将预测概率四舍五入到最近的整数来得到预测的类别
    pred = torch.sigmoid(pred)
    pred_rounded = torch.round(pred)
    
    # 比较预测的类别和真实的类别
    correct_predictions = (pred_rounded == target)
    
    # 计算准确率
    accuracy = correct_predictions.sum().float() / correct_predictions.numel()
    
    return accuracy.item()

def atom_iou(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()  
    union = pred.sum() + target.sum() - intersection.sum()
    iou = (intersection + 1e-7) / (union + 1e-7)
    soft_iou_loss = 1 - iou.mean()

    return soft_iou_loss

def batch_atom_iou(pred, target, lengths):
    batch_size = pred.size(0)
    total_iou = 0.0

    for i in range(batch_size):
        length = lengths[i]
        pred_seq = pred[i, :length]
        target_seq = target[i, :length]
        if pred_seq.shape[0]>target_seq.shape[0]:
            pred_seq=pred_seq[:target_seq.shape[0]]
        pred_seq = torch.sigmoid(pred_seq)
        intersection = (pred_seq * target_seq).sum()
        union = pred_seq.sum() + target_seq.sum() - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)

        total_iou += iou
    mean_iou = total_iou / batch_size
    soft_iou_loss = 1 - mean_iou

    return soft_iou_loss


def calculate_accuracy_cls10(pred, target):
    """
    计算预测准确率。
    参数:
        pred (torch.Tensor): 模型输出的预测概率。
        target (torch.Tensor): 真实的目标标签。
    返回:
        accuracy (float): 预测的准确率。
    """
    _, pred_classes = torch.max(pred, dim=1)
    
    # 比较预测的类别和真实的类别
    correct_predictions = (pred_classes == target)
    
    # 计算准确率
    accuracy = correct_predictions.sum().float() / target.size(0)
    
    return accuracy.item()

import scipy.stats
def calculate_spearman(pred, target):
    # 将PyTorch张量转换为NumPy数组
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # 计算Spearman's ρ
    rho, _ = scipy.stats.spearmanr(pred_np, target_np)
    
    return rho

def save_checkpoint(epoch, model, optimizer, save_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, save_dir)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# from transformers import AutoTokenizer, EsmModel, EsmTokenizer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
class MLPs(nn.Module):
    def __init__(self,in_dim, hidden_dim, out_dim, layer_num):
        super(MLPs, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i != 0 and i != len(self.layers)-1:
                hidden = layer(x) + x
                x = hidden
                x = self.activation(x)
                x = self.dropout(x)
            else:
                x = layer(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.query = nn.Linear(d_model, d_model)
        # self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model) 
        
    def forward(self, embedding_output, structure):
        v = self.value(embedding_output)
        attn_weights = structure

        context = torch.matmul(attn_weights, v)
        out = embedding_output + context
        out = self.norm(out)
        return out

class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Fusion(nn.Module):
    def __init__(self, feat_dim = 1280, mid_dim = 768):
        super(Fusion, self).__init__()
        self.activation = torch.nn.GELU()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(16, mid_dim)
        self.fc2 = nn.Linear(16, mid_dim)
        self.fc3 = nn.Linear(mid_dim, self.feat_dim)

    def forward(self, sidi, badi, tau, theta):
        bs, n = sidi.shape[:2]
        tau = tau.unsqueeze(-1).float()
        theta = theta.unsqueeze(-1).float()
        sidi = sidi.float()
        badi = badi.float()

        angle = torch.cat([badi, sidi, tau, theta], dim=-1)

        sin_values = torch.sin(angle)
        cos_values = torch.cos(angle)
        x = torch.cat((sin_values, cos_values), dim=-1)
        x1 = self.activation(self.fc(x))
        x2 = self.fc2(x)
        x = self.fc3(x1*x2)
        return x
    
#plddt ver
class Fusion_plddt(nn.Module):
    def __init__(self, feat_dim = 1280, mid_dim = 768):
        super(Fusion_plddt, self).__init__()
        self.activation = torch.nn.GELU()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(17, mid_dim)
        self.fc2 = nn.Linear(17, mid_dim)
        self.fc3 = nn.Linear(mid_dim, self.feat_dim)

    def forward(self, sidi, badi, tau, theta, plddt):
        bs, n = sidi.shape[:2]
        tau = tau.unsqueeze(-1).float()
        theta = theta.unsqueeze(-1).float()
        sidi = sidi.float()
        badi = badi.float()
        plddt = plddt.unsqueeze(-1).float()
        plddt = plddt/100
        angle = torch.cat([badi, sidi, tau, theta], dim=-1)

        sin_values = torch.sin(angle)
        cos_values = torch.cos(angle)
        x = torch.cat((sin_values, cos_values, plddt), dim=-1)
        x1 = self.activation(self.fc(x))
        x2 = self.fc2(x)
        x = self.fc3(x1*x2)
        return x