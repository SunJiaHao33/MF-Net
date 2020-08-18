import os
import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F 
from torchvision import models

from network.customize import StripPooling, dirpooling



nonlinearity = partial(F.relu, inplace=True)



class Fusionpooling(nn.Module):
    def __init__(self, in_channels):
        super(Fusionpooling, self).__init__()
        
        #最大值池化
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)#7*7
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)#4*4
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)#2*2
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)#2*2
        
        #线性池化
        self.pool5 = StripPooling(in_channels, 2)

        #平均池化

        self.pool6 = nn.AvgPool2d(kernel_size=[2, 2], stride=2)
        self.pool7 = nn.AvgPool2d(kernel_size=[6, 6], stride=6)



        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.relu_(F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer2 = F.relu_(F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer3 = F.relu_(F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer4 = F.relu_(F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True))
        

        self.layer5 = self.pool5(x)

        self.layer6 = F.relu_(F.interpolate(self.conv(self.pool6(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer7 = F.relu_(F.interpolate(self.conv(self.pool7(x)), size=(h, w), mode='bilinear', align_corners=True))

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, x], 1)

        return out
class Fusionpooling_withoutx(nn.Module):
    def __init__(self, in_channels):
        super(Fusionpooling_withoutx, self).__init__()
        
        #最大值池化
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)#7*7
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)#4*4
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)#2*2
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)#2*2
        
        #线性池化
        self.pool5 = StripPooling(in_channels, 2)

        #平均池化

        self.pool6 = nn.AvgPool2d(kernel_size=[2, 2], stride=2)
        self.pool7 = nn.AvgPool2d(kernel_size=[6, 6], stride=6)



        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.relu_(F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer2 = F.relu_(F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer3 = F.relu_(F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer4 = F.relu_(F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True))
        

        self.layer5 = self.pool5(x)

        self.layer6 = F.relu_(F.interpolate(self.conv(self.pool6(x)), size=(h, w), mode='bilinear', align_corners=True))
        self.layer7 = F.relu_(F.interpolate(self.conv(self.pool7(x)), size=(h, w), mode='bilinear', align_corners=True))

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7], 1)

        return out

#固定核卷积
class Fixedconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fixedconv, self).__init__()
        #固定核卷积
        self.conv1 = dirpooling(in_channels, out_channels)
        #线性卷积
        self.conv2 = nn.Conv2d(in_channels, 1, (1,3), 1, (0,1))
        self.conv3 = nn.Conv2d(in_channels, 1, (3,1), 1, (1,0))
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = self.conv2(x)
        x2 = self.bn(x2)
        x3 = self.conv3(x)
        x3 = self.bn(x3)

        out = torch.cat([x1, x2, x3], dim=1)
        out = F.relu_(out)

        return out

# 解码器
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Net_34(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_34, self).__init__()
        self.name = 'Net_34'

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1 # 7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 池化+固定核卷积
        
        self.Fpblock = Fusionpooling(512) # 512->520
        self.Fcblock = Fixedconv(512, 6) # 512->8


        self.decoder4 = DecoderBlock(528, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        e5 = self.Fpblock(e4)
        e4 = self.Fcblock(e4)
        
        e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


class Net_50(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_50, self).__init__()
        self.name = 'Net_50'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1#7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #池化+固定核卷积
        
        self.Fpblock = Fusionpooling(2048)#2048->2056
        self.Fcblock = Fixedconv(2048,6)#2048->8


        self.decoder4 = DecoderBlock(2064, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        e5 = self.Fpblock(e4)
        e4 = self.Fcblock(e4)
        
        e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class Net_50_add(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_50_add, self).__init__()
        self.name = 'Net_50_add'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1#7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #池化+固定核卷积
        
        self.Fpblock = Fusionpooling_withoutx(2048)#2048->2056
        self.Fcblock = Fixedconv(2048,6)#2048->8


        self.decoder4 = DecoderBlock(2056, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        e5 = self.Fpblock(e4)
        e6 = self.Fcblock(e4)
        
        e4 = torch.cat([e4, e5+e6], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class Net_50_FCK(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_50_FCK, self).__init__()
        self.name = 'Net_50_FCK'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1#7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #池化+固定核卷积
       
        # self.Fpblock = Fusionpooling(2048)#+8
        self.Fcblock = Fixedconv(2048,6)#->8


        self.decoder4 = DecoderBlock(2056, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        # e5 = self.Fpblock(e4)
        e5 = self.Fcblock(e4)
        
        e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class Net_50_MFP(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_50_MFP, self).__init__()
        self.name = 'Net_50_MFP'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1#7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #池化+固定核卷积
       
        self.Fpblock = Fusionpooling(2048)#2048->2056
        # self.Fcblock = Fixedconv(2048,6)#->6


        self.decoder4 = DecoderBlock(2056, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        e4 = self.Fpblock(e4)
        # e5 = self.Fcblock(e4)
        
        # e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class Net_50_baseline(nn.Module):
    def __init__(self,num_classes=1, num_channels=3):
        super(Net_50_baseline, self).__init__()
        self.name = 'Net_50_baseline'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1#7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #池化+固定核卷积
       
        # self.Fpblock = Fusionpooling(2048)#2048->2056
        # self.Fcblock = Fixedconv(2048,6)#->6


        self.decoder4 = DecoderBlock(2048, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        
        # e5 = self.Fpblock(e4)
        # e5 = self.Fcblock(e4)
        
        # e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class Net_101(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Net_101, self).__init__()
        self.name = 'Net_101'
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1  # 7x7 2 3
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 池化+固定核卷积
        # self.dblock = DACblock(512)
        self.Fpblock = Fusionpooling(2048)  # 512->520
        self.Fcblock = Fixedconv(2048, 6)  # 512->8

        self.decoder4 = DecoderBlock(2064, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        # e4 = self.dblock(e4)
        e5 = self.Fpblock(e4)
        e4 = self.Fcblock(e4)

        e4 = torch.cat([e4, e5], dim=1)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


if __name__=="__main__":
    print(os.path)
    a = torch.rand(1, 3, 61, 570)
    model = Net_34()
    
    # with SummaryWriter(comment='Net_34') as w:
    #     w.add_graph(model, (a, ))
    b = model(a)
    print(b.size())