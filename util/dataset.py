
from PIL import Image
from torch.utils.data import Dataset
from util.tool import compute_grad
import torch


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # str.rstrip([chars])删除指定字符，默认空格
            words = line.split()  # str.split(str="", num=string.count(str)).
            imgs.append((words[0], int(words[1]), words[2])) # word[2是什么]
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.root = "./pascal_train_set/JPEGImages/"

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        path, lam, th = self.imgs[index]
        img = Image.open(self.root + path).convert('RGB')  # 讲img转换成RGB
        if self.transform is not None:  # 是否进行transform
            img = self.transform(img)  #遍历transform中的函数 按顺序应用到img中
        lam = torch.full((1, img.size()[1], img.size()[2]), lam)  #行列 我是全为1的输入
        img_grad_x, img_grad_y = compute_grad(img, float(th))
        data = torch.cat((img, img_grad_x, img_grad_y, lam), dim=0)  # 拼接输入的通道。

        return data

    def __len__(self):
        return len(self.imgs)
