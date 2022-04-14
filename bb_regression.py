# 利用最后一个池化层pool输出的特征去训练一个线性回归模型，以预测新的检测框
import utils
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch import nn


def train(net, epochs, lr, criterion, device, transform):
    class Reg_dataset(Dataset):
        def __init__(self, ss_csv_path, gt_csv_path, transform, net, device):
            self.ss_csv = pd.read_csv(ss_csv_path)
            self.gt_csv = pd.read_csv(gt_csv_path)
            self.transform = transform
            self.net = net
            self.device = device
        
        def __getitem__(self, index):
            img_path, *ss_loc = self.ss_csv.iloc[index, :]
            index = img_path.split('/')[-1].split('_')[0]+'.png'
            gt_loc = self.gt_csv[self.gt_csv.img_name==index].iloc[0, 2:].tolist()
            label = torch.tensor(gt_loc, dtype=torch.float32) - torch.tensor(ss_loc, dtype=torch.float32)
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                img = self.transform(img).to(device).unsqueeze(0)
                return net.features(img).squeeze(0), label
        
        def __len__(self):
            return len(self.ss_csv)
    
    ss_train_loc = './data/ss_train_loc.csv'
    gt_train_loc = './data/banana-detection/bananas_train/label.csv'
    ss_val_loc = './data/ss_val_loc.csv'
    gt_val_loc = './data/banana-detection/bananas_val/label.csv'

    train_data = Reg_dataset(ss_train_loc, gt_train_loc, transform, net=net, device=device)
    val_data = Reg_dataset(ss_val_loc, gt_val_loc, transform, net=net, device=device)
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=128)

    linear_net = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256*6*6, 4))
    nn.init.xavier_normal_(linear_net[-1].weight)
    utils.train(train_dataloader, val_dataloader, linear_net, epochs, lr, criterion, device, len(train_data), len(val_data), mode='regression')

