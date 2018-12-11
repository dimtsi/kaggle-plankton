import torch
import torch.nn as nn
import math;
from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader,TensorDataset
from torch.autograd import Variable
from torchvision import transforms
import math
import glob
import cv2

from torchsummary import summary
from Preprocessing import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Fusion(nn.Module):
    def forward(self, input):
        for i, single_tensor in enumerate(input):
            if i==0:
                concat = input[i]
            else:
                concat = torch.cat((concat, input[i]), dim = 1)
        return concat

class ResNetMine(nn.Module):

    def __init__(self, block, layers, num_classes=121):
        self.inplanes = 64
        super(ResNetMine, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        print(self.layer1.state_dict())
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            # nn.Linear(128*block.expansion, 64*block.expansion),
            # nn.LeakyReLU(0.3),
            nn.Dropout(0.3)
            )


        self.fc2 = nn.Linear(128*block.expansion, num_classes)
        ##

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


class ResNetDynamic(nn.Module):

    def __init__(self, block, layers, num_classes=121, num_layers=4, pretrained_nn = None):
        self.inplanes = 64
        self.num_layers = num_layers
        self.layers = layers
        self.block = block
        super(type(self), self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        inside_layers = OrderedDict()
        layer_planes = 64
        for i in range(num_layers):
            if i==0:
                inside_layers[str(i)] = self._make_layer(block, layer_planes, layers[i])
            else:
                inside_layers[str(i)] = self._make_layer(block, layer_planes, layers[i], stride=2)
            layer_planes *= 2
        self.inside_layers = nn.Sequential(inside_layers)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = Flatten()
        self.final_size = 64* block.expansion * 2**(num_layers-1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.final_size, self.final_size),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.6)
            )
        self.fc2 = nn.Linear(self.final_size, num_classes)
        ##

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if (pretrained_nn != None):
            self.pretrain(pretrained_nn)

    def pretrain(self, pretrained_resnet): ###Only for use with original ResNet
        pretrained_layers = []
        pretrained_layers.append(pretrained_resnet.layer1)
        pretrained_layers.append(pretrained_resnet.layer2)
        pretrained_layers.append(pretrained_resnet.layer3)
        pretrained_layers.append(pretrained_resnet.layer4)

        for i in range(self.num_layers):
            self.inside_layers[i].load_state_dict(pretrained_layers[i].state_dict())


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.inside_layers(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class FeatureBoostedCNN(nn.Module):
     def __init__(self, network, num_extra_feats=0, num_classes=121):
        super(type(self), self).__init__()
        self.convolutional =  nn.Sequential(*list(network.children())[:-1])#[:-2]
        self.cnn_final_size =  64* network.block.expansion * 2**(network.num_layers-1)
        self.flattened_size = 256 + num_extra_feats
        self.flatten = Flatten()
        self.fusion=Fusion()
        self.fc1 = nn.Sequential(
            # nn.Linear(self.cnn_final_size, self.cnn_final_size//2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4)
            )
        self.fc2 = nn.Linear(self.flattened_size, num_classes)

     def forward(self, x):
        x1 = self.convolutional(x[0])
        x1 = self.fusion([x1, x[1]])
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        return x1

class SuperNet(nn.Module):

    def __init__(self, networks, num_classes=121):
        super(type(self), self).__init__()
        self.net1 =  nn.Sequential(*list(networks[0].children())[:-1])
        self.net2 =  nn.Sequential(*list(networks[1].children())[:-1])

        self.final_size = 0
        for net in networks:
            self.final_size += net.final_size//2

        self.fc = nn.Linear(self.final_size, num_classes)

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return z

class EnsembleClassifier(nn.Module):

    def __init__(self, networks, num_classes=121, multiGPU = False):
        self.devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2"), torch.device("cuda:3")]
        self.multiGPU = multiGPU
        super(type(self), self).__init__()
        self.net1 =  nn.Sequential(*list(networks[0].children()))
        self.net1.requires_grad = False
        self.net2 =  nn.Sequential(*list(networks[1].children()))#[:-1]
        self.net2.requires_grad = False
        self.net3 =  nn.Sequential(*list(networks[2].children()))#[:-1]
        self.net3.requires_grad = False
        self.net4 =  nn.Sequential(*list(networks[3].children()))#[:-1]
        self.net4.requires_grad = False

        self.fusion = Fusion()
        self.final_size = 0
        for net in networks:
            self.final_size += num_classes
        self.fc1 = nn.Sequential(
            nn.Linear(self.final_size, (int)(1.2*self.final_size)),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.6)
            )
        self.fc2 = nn.Linear((int)(1.2*self.final_size), num_classes)
#
    def forward(self, x):
        if self.multiGPU == True:
            x1 = self.net1(x.to(self.devices[0]))
            x2 = self.net2(x.to(self.devices[1]))
            x3 = self.net3(x.to(self.devices[2]))
            x4 = self.net4(x.to(self.devices[3]))
            z = self.fusion([x1, x2.to(self.devices[0]),
                            x3.to(self.devices[0]),
                            x4.to(self.devices[0])])

        else:
            x1 = self.net1(x)
            x2 = self.net2(x)
            x3 = self.net3(x)
            x4 = self.net4(x)
            z = self.fusion([x1, x2.to(self.devices[0]),
                             x3.to(self.devices[0]),
                             x4.to(self.devices[0])])#

        z = self.fc1(z)
        z = self.fc2(z)
        return z

    def set_devices_multiGPU(self):
        self.multiGPU = True
        self.net1.to(self.devices[0])
        self.net2.to(self.devices[1])
        self.net3.to(self.devices[2])
        self.net4.to(self.devices[3])

class PretrainedResnetMine(ResNetMine):

    def __init__(self, block, layers, num_classes=121, pretrained_nn = None):
        super(type(self), self).__init__(block, layers)
        self.layer1.load_state_dict(pretrained_nn.layer1.state_dict())
        self.layer2.load_state_dict(pretrained_nn.layer2.state_dict())
        print(self.layer1.state_dict())
        # self.layer3.load_state_dict(pretrained_nn.layer3.state_dict())
        # self.layer4.load_state_dict(pretrained_nn.layer4.state_dict())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(512 * 4, num_classes)



class CNN(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            Flatten(),
            nn.Linear(20*16*16, 250),
            nn.ReLU())
        self.fc = nn.Linear(250, 121)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        return out


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(type(self), self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(type(self), self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(type(self), self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.block = block
        self.layers = layers

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
