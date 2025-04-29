from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch

def parse_args():
    #argparse模块的作用是用于解析命令行参数。
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')

    #向该对象中添加命令行参数和选项
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    #解析添加的参数
    args = parser.parse_args()
    return args

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    对数据进行预处理
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        #transforms.ToTensor()为转换为tensor格式，transforms.Normalize为对像素值进行归一化处理
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize(mean=(0.485, 0.456, 0.406) , std=(0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)  #把数组转换成张量
    return img


if __name__ == '__main__':

    args = parse_args()

    # create depth model，创建模型
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint，加载模型参数
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

    #文件夹路径
    image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    #用于返回指定的文件夹包含的文件或文件夹的名字的列表
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    #os.path.join为文件路径拼接，路径为文件夹image_dir+文件名i
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    #输出文件夹
    image_dir_out = image_dir + '/outputs'
    #创建文件夹image_dir_out
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)

        #实现RGB到BGR通道的转换（rgb[:, :, ::-1]）
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None

        #改变输出图像尺寸大小(448, 448)
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        #对图像进行处理，转化为张量
        img_torch = scale_torch(A_resize)[None, :, :, :]
        #得到预测深度值，squeeze()维度压缩，在0起的指定位置N，去掉维数为1的的维度
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        #将预测深度调整到原始图像大小
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)
        #深度图对应的尺度因子
        img_name = v.split('/')[-1]
        cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
