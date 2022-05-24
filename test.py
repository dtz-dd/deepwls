import visdom
from net.fcnBn import FcnBn
import torch
from PIL import Image
from torchvision import transforms
# from util.tool import img_grad
import time
import os
from tensorboardX import SummaryWriter

def main(device):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    file_name = "320X240"
    # change filename
    folder_path = "D:/labdata/ImageSmoothing-main/predict_set/60dataset/deepwls/smooth/" + file_name +"/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    model = FcnBn()
    model.load_state_dict(torch.load(
        './model/smoothing_gs_constantvalue0.15_10.pth'))
    # model.load_state_dict(torch.load('E:/wudan/store1/smoothing_gs_constantvalue0.1_10.pth'))
    if device == 'gpu':
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = torch.device('cpu')
    dir1 = "D:/labdata/groundtruth/640X480/"
    # dir1 = "D:/labdata/groundtruth/groundtruth/"
    # dir1 = "D:/labdata/groundtruth/testSet/"
    # dir2 = "./predict_set/new2/"
    a = 1
    filelist = []
    timeall = 0
    filenames = os.listdir(dir1)
    # filenames.sort(key=lambda x:int(x[:-4])) 
    # print(filenames)
    for fn in filenames:
        if fn == '.DS_Store':
            continue
        fullfilename = os.path.join(dir1, fn)
        #filelist.append(fullfilename)
        print(fn)
        img = Image.open(fullfilename).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        x = img
        with torch.no_grad():
            model.eval()
            torch.cuda.synchronize()
            time_1 = time.time()
            y = model(x)
            torch.cuda.synchronize()
            time_2 = time.time()
            timed = (time_2 - time_1)
            print(timed, " s")
            timeall += timed
            print(timeall)
        y = y.cpu()
        # new_img_PIL = transforms.ToPILImage()(y[0])
        
        # new_img_PIL.save(folder_path + 'a' + fn)
        # new_img_PIL.save(folder_path + fn + ".png")
        # a = a+1


if __name__ == "__main__":
    # evaluateWithGpu()
    main('gpu')
 
