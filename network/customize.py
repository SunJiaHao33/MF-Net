import torch
import torch.nn as nn
from torch.nn import functional as F 

#线性平均池化
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, out_channels):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1xn
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#nx1

        inter_channels = in_channels//4
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))#1x1卷积+激活
       
         
        self.conv1_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))#1x3卷积
        self.conv3_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))#3x1卷积
        
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1,1,0, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))#1x1卷积+激活

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv1(x)#channel/4
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x1 = self.conv1_3(x1)
        x2 = self.conv3_1(x2)
        x1 = F.interpolate(x1, (h,w),  mode='bilinear',align_corners=True)
        x2 = F.interpolate(x2, (h,w),  mode='bilinear',align_corners=True)
        out = F.relu_(x1 + x2)
        out = self.conv2(out)
        return out

#方向池化自定义滤波器

class dirpooling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(dirpooling,self).__init__()
        self.channels = in_channels

        filter1 = torch.tensor([[1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]])
        kernel1 = torch.Tensor(filter1).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel1 = kernel1.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight1 = nn.Parameter(data=kernel1, requires_grad=False)

        filter2 = torch.tensor([[0., 0., 1.],
                                [0., 1., 0.],
                                [1., 0., 0.]])
        kernel2 = torch.Tensor(filter2).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel2 = kernel2.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        filter3 = torch.tensor([[1., 0., 1.],
                                [0., 1., 0.],
                                [1., 0., 1.]])
        kernel3 = torch.Tensor(filter3).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel3 = kernel3.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight3 = nn.Parameter(data=kernel3, requires_grad=False)

        filter4 = torch.tensor([[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]])
        kernel4 = torch.Tensor(filter4).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel4 = kernel4.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight4 = nn.Parameter(data=kernel4, requires_grad=False)
        
        filter5 = torch.tensor([[0., 0., 0.],
                                [1., 1., 1.],
                                [0., 0., 0.]])
        kernel5 = torch.Tensor(filter5).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel5 = kernel5.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight5 = nn.Parameter(data=kernel5, requires_grad=False)

        filter6 = torch.tensor([[0., 1., 0.],
                                [0., 1., 0.],
                                [0., 1., 0.]])
        kernel6 = torch.Tensor(filter6).unsqueeze(0).unsqueeze(0)    # (3, 3) -> (1, 1, 3, 3)
        kernel6 = kernel6.expand((out_channels//6, int(self.channels), 3, 3))
        self.weight6 = nn.Parameter(data=kernel6, requires_grad=False)

        
    def forward(self,x):
        b, c, h, w =x.size()
        x1 = F.conv2d(x, self.weight1, padding=1,stride=1)
        x2 = F.conv2d(x, self.weight2, padding=1,stride=1)
        x3 = F.conv2d(x, self.weight3, padding=1,stride=1)
        x4 = F.conv2d(x, self.weight4, padding=1,stride=1)
        x5 = F.conv2d(x, self.weight5, padding=1,stride=1)
        x6 = F.conv2d(x, self.weight6, padding=1,stride=1)
        out = torch.cat([x1, x2, x3, x4, x5, x6],dim = 1)
        out = F.relu_(out)
        return out

       
if __name__ == "__main__":
    a = torch.randn(1,64,14,14)

    c = torch.tensor([[[[1.,3.,2.],
                        [2.,3.,1.],
                        [3.,1.,2.]]]])
    # c = torch.randn(3,3)
    # c = torch.Tensor(c).unsqueeze(0).unsqueeze(0)
    print(c)
    print(c.size())
    # model = StripPooling(64,1)
    # print(model.parameters)
    # b = model(a)
    model1 = dirpooling(1,1)
    b = model1(c)
    print(model1.parameters)
    print(b)
    
    
    print(b.size())  
    # print(b.size())