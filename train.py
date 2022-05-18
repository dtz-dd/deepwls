import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import visdom
from net.fcnBn import FcnBn
from util.tool import produce_lamd, img_grad, load_data, produce_apha, img_gradient, guiyi, weight
from util.dataset import MyDataset
import math
import random
from torch import nn
import time
from loss_function.new_loss import new_loss
from tensorboardX import SummaryWriter


def initial_parameters(model):
    """
    初始化网络参数
    :param model: 网络
    """

    for idx, m in enumerate(model.modules()):  # enumerate
        if isinstance(m, nn.Conv2d):
            size = m.weight.shape
            stdv = math.sqrt(
                12/(size[1]*size[2]*size[3] + size[0]*size[2]*size[3]))
            print(stdv)
            torch.nn.init.uniform_(m.weight, a=-stdv, b=stdv)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def get_data():
    """
    获取训练集
    """
    data = MyDataset("./processed_data/data.txt", transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))
    train_set = DataLoader(data, batch_size=16, shuffle=True)
    return train_set


# @torchsnooper.snoop()
def main():
    # 获取训练集
    train_set = load_data("D:/ImageSmoothing/ImageSmoothing/pascal_train_set", 16)
    device = torch.device('cuda')
    model = FcnBn()  # 网络
    # model.load_state_dict(torch.load('./model/newloss/90smoothing_gs_constantvalue0.03_5000.pth'))  # 把加载的权重复制到模型的权重中去

    criterion = new_loss()  # 改成想要的loss

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # lr学习率
    model = model.to(device)
    # 初始化可视化工具
    writer = SummaryWriter()
    x, _ = iter(train_set).next()
    # 绘制网络结构
    height = x.size()[2]
    width = x.size()[3]
    # 网络训练
    # min_loss = 1
    time1 = time.time()

    for epoch in range(1, 120):  
        running_loss = 0
        for batchidx, (imgs, _) in enumerate(train_set):


            imgs = imgs.to(device)

            x = imgs
            lamd = 10 #Modified training parameters
            alpha = 0.08
            x_hat = model(x)
            loss = criterion(x, x_hat, lamd, alpha)  # 这里调用

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "./model/newloss/{}smoothing_gs_constantvalue{}_{}.pth".format(epoch,lamd,alpha))
        m_loss = running_loss / len(train_set)

        print(epoch, 'train_loss:', m_loss)

        writer.add_scalar('train_loss', m_loss, epoch)
        writer.add_images(tag='train_input',
                          img_tensor=imgs, global_step=epoch)
        writer.add_images(tag='train_output',
                          img_tensor=x_hat, global_step=epoch)


    time2 = time.time()
    print((time2 - time1)/60/60, " 小时")
    # 保存训练好的模型
    # save(module,'')保存整个模型；
    torch.save(model.state_dict(),
               "./model/newloss/smoothing_gs_constantvalue{}_{}.pth".format(lamd,alpha))
    # save(model.state_dict(),'')保存训练好的权重


if __name__ == '__main__':
    main()
