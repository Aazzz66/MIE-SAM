import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import itertools
import pandas as pd
from torch import nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from skimage import io
import os
from data import SalObjDataset, get_loader

WINDOW_SIZE = (256, 256)

STRIDE = 32
IN_CHANNELS = 3

train_root = './dataest/VT5000/Train'
test_root = './dataest/VT5000/Test'
#test_root = './dataest/VT821'
#test_root = './dataest/VT1000'
BATCH_SIZE = 5
trainsize = 256
train_loader = get_loader(train_root=train_root, batchsize=BATCH_SIZE, trainsize=trainsize)

N_CLASSES = 2
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True
MODEL = 'MIESAM'
MODE = 'Train'
#MODE = 'Test'
DATASET = 'VT5000'
#DATASET = 'VT821'
IF_SAM = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
Stride_Size = 32
epochs = 100
save_epoch = 1

print(MODEL + ', ' + MODE + ', ' + DATASET + ', IF_SAM: ' + str(IF_SAM) + ', WINDOW_SIZE: ', WINDOW_SIZE,
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)


def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def LossFunc(pred, mask):

    mask=mask.unsqueeze(1).float()

   # print('lossfunc pred',pred.shape,'mask',mask.shape)

    bce = F.binary_cross_entropy(pred, mask, reduction='mean')
    #print('bce', bce)

    inter = ((pred * mask)).sum(dim=(2, 3))
    #print('inter',inter)
    union = ((pred + mask)).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)
    aiou1 =aiou.mean()
    #print('aiou1',aiou1)
    mae = F.l1_loss(pred, mask, reduction='mean')
    #loss = bce + aiou1 + mae
    #loss1 = (bce + aiou1 + mae).mean()
    #print('loss',loss,'loss1',loss1)
    return bce , aiou1 , mae



def accuracy(input, target):
    if isinstance(input, torch.Tensor):
        input = input.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    if target.size == 0:
        return 0
    return 100 * float(np.count_nonzero(input == target)) / target.size



def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def grouper(n, iterable):
    """将迭代器分组为n个元素的元组"""
    args = [iter(iterable)] * n
    return zip(*args)

from sklearn.metrics import confusion_matrix, f1_score, recall_score

def convert_to_grayscale(arr_2d):
    """将2D数组转换为灰度图像"""
    return arr_2d.astype(np.uint8)

def normalize_to_unit_interval(data):
    """将数据线性归一化到[0,1]区间"""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)


def metrics(pred, mask):

    pred= torch.from_numpy(pred).float()
    mask= torch.from_numpy(mask).float()

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    inter = ((pred * mask)).sum()
    union = ((pred + mask)).sum()
    aiou = 1 - (inter + 1) / (union - inter + 1)
    mae = F.l1_loss(pred, mask, reduce=None)
    return bce.mean(),aiou.mean(),mae.mean()

def plot_loss_curves(train_losses, train_metrics, val_losses, val_metrics, lr_history=None, save_path=None):
    """
    绘制带自动异常值处理的训练曲线
    Args:
        train_losses: 训练loss列表/数组
        train_metrics: 训练指标字典（如 {'bce': [], 'aiou1': [], 'mae': []}）
        val_losses: 验证loss列表/数组
        val_metrics: 验证指标字典（格式同 train_metrics）
        lr_history: 学习率历史记录（可选）
        save_path: 图片保存路径（可选）
    """
    def _process_data(data):
        if isinstance(data, dict):
            return {k: pd.Series(np.array(v, dtype=np.float32)).interpolate().values for k, v in data.items()}
        else:
            data = np.array(data, dtype=np.float32)
            data[~np.isfinite(data)] = np.nan
            return pd.Series(data).interpolate().values

    train_losses = _process_data(train_losses)
    val_losses = _process_data(val_losses)
    train_metrics = _process_data(train_metrics)
    val_metrics = _process_data(val_metrics)

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(train_losses, 'b-', alpha=0.7, linewidth=2)
    plt.title('Training Loss (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(3, 2, 3)
    plt.plot(val_losses, 'r-', alpha=0.7, linewidth=2)
    best_epoch = np.argmin(val_losses)
    plt.scatter(best_epoch, val_losses[best_epoch], c='red', s=50,
                label=f'Best Epoch: {best_epoch}')
    plt.title('Validation Loss (Best Epoch Marked)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(3, 2, 4)
    for metric, color in zip(['bce', 'aiou1', 'mae'], ['r', 'g', 'b']):
        plt.plot(train_metrics[metric], f'{color}-', label=f'Train {metric}', alpha=0.5)
    plt.title('Metrics Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Train')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(3, 2, 5)
    for metric, color in zip(['bce', 'aiou1', 'mae'], ['r', 'g', 'b']):
        plt.plot(val_metrics[metric], f'{color}--', label=f'Val {metric}', alpha=0.8)
    plt.title('Metrics Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()


    if lr_history is not None:
        plt.subplot(3, 2, 6)
        plt.plot(lr_history, 'm-', label='Learning Rate', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Curves saved to: {save_path}")


