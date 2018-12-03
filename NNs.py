import torch
import torch.nn as nn
import math;

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(0.3, inplace=True)
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

class SuperNet(nn.Module):

    def __init__(self, networks, num_classes=121):
        super(SuperNet, self).__init__()
        self.net1 =  nn.Sequential(*list(networks[0].children())[:-1]).cuda()
        self.net2 =  nn.Sequential(*list(networks[1].children())[:-1]).to(torch.device('cuda:2'))
        self.fc = nn.Linear(128*4*2, num_classes)
        # if torch.cuda.device_count() > 1:
        #     print("2GPU")
        #     self.net1.cuda(device = gpus[0])
        #     self.net2.cuda(device = gpus[3])
            # print(self.net2.device())



    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x.to(torch.device('cuda:2')))
        z = torch.cat((x1, x2.to(torch.device('cuda:0'))), 1)
        z = z.view(z.size(0), -1)
        z = self.fc(z)
        return z
#         print(x.size())

#         x = self.relu(x)
#         print(x.size())
#         x = self.maxpool(x)
#         print(x.size())

#         x = self.layer1(x)
#         print(x.size())

#         x = self.layer2(x)
#         print(x.size())
#         x = self.layer3(x)
#         print(x.size())
#         x = self.layer4(x)
#         print(x.size())
#         x = self.avgpool(x)
#         print(x.size())
#         x = x.view(x.size(0), -1)
#         print(x.size())

#         x = self.fc(x)

#         return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
