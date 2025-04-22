from lib import network_auxi as network
from lib.net_tools import get_func
import torch
import torch.nn as nn

class RelDepthModel(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(RelDepthModel, self).__init__()
        #backbone主干网络
        if backbone == 'resnet50':
            #编码器为resnet50
            encoder = 'resnet50_stride32'
        elif backbone == 'resnext101':
            encoder = 'resnext101_stride32x8d'
        self.depth_model = DepthModel(encoder)

    #深度预测
    def inference(self, rgb):
        with torch.no_grad():
            input = rgb.cuda()
            depth = self.depth_model(input)
            pred_depth_out = depth - depth.min() + 0.01
            return pred_depth_out


class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()

        backbone = network.__name__.split('.')[-1] + '.' + encoder
        #帮助程序按名称返回函数对象,相当于self.encoder_modules = resnet50_stride32()
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit