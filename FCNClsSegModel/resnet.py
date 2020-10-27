import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride) 
        self.bn1 = nn.BatchNorm2d(planes) # bn
        self.relu = nn.ReLU(inplace=True) # relu
        self.conv2 = conv3x3(planes, planes) # conv3x3
        self.bn2 = nn.BatchNorm2d(planes) # BN
        self.downsample = downsample # 
        self.stride = stride

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes) # 
        self.bn1 = nn.BatchNorm2d(planes) # BN
        self.conv2 = conv3x3(planes, planes, stride) # 3x3
        self.bn2 = nn.BatchNorm2d(planes) # BN
        self.conv3 = conv1x1(planes, planes * self.expansion) # 
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) # BN
        self.relu = nn.ReLU(inplace=True) # 
        self.downsample = downsample # downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, input_shapes=(3, 224, 244)):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_shapes[0], 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def save_gradient(self, grad):
        alpha_c = grad.mean(dim=(-2,-1)) # mean over space slice, batch_size x channels
        self.alpha.append(alpha_c.unsqueeze(-1))  

    def clear_gradient(self):
        self.alpha = []


    def sethook(self):
        for dense_feature in self.dense_features:
            dense_feature.register_hook(self.save_gradient)

    def get_cam(self):
        #features = F.relu(self.dense_features[-1])
        if self.fc.bias.shape[-1]==1:
            features = self.dense_features[-1]
            batch_size, channels, height, width = features.shape
            
            features = features.view((batch_size, channels, height*width)).transpose(1,2).view((batch_size*height*width, channels))
            #print(features.shape)
            cam = self.fc(features)
            #print(cam.shape)
            cam = cam.transpose(0,1).view((batch_size, 1, height, width))
            cam = cam-self.fc.bias[-1]
        else:
            features = self.dense_features[-1]
            batch_size, channels, height, width = features.shape
            
            features = features.view((batch_size, channels, height*width)).transpose(1,2).view((batch_size*height*width, channels))
            #print(features.shape)
            cam = torch.softmax(self.fc(features),-1)[:,-1:]
            #print(cam.shape)
            cam = cam.transpose(0,1).view((batch_size, 1, height, width))
            #cam = cam-self.fc.bias[-1]

        return cam

    def get_gradcam(self, cls_score):
        grad_cam_blocks = []

        mean_class_y = cls_score.sum(dim=0)[-1:] # classes_num

        for y in mean_class_y: 
            self.clear_gradient() 
            grad_cam_block = [] 
            self.sethook() 
            y.backward(retain_graph=True) 
            # 
            for b, alpha in enumerate(self.alpha, 1): 
                grad_cam_block.append(torch.sum(alpha.unsqueeze(-1)*self.dense_features[-b], dim=1, keepdim=True))

            grad_cam_blocks.append(grad_cam_block)

            
            grad_cam_blocks = [torch.cat([grad_cam_blocks[c][b] for c in range(len(mean_class_y))], 1) for b in range(len(grad_cam_blocks[0])-1, -1, -1)]
        return grad_cam_blocks


    def forward(self, x, genCAM=False):
        if genCAM:
            self.dense_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if genCAM:
            self.dense_features.append(x)

        x = self.layer2(x)
        if genCAM:
            self.dense_features.append(x)

        x = self.layer3(x)
        if genCAM:
            self.dense_features.append(x)

        x = self.layer4(x)
        if genCAM:
            self.dense_features.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if genCAM:
            cam = self.get_cam()
            if x.shape[-1] == 1:
                cls_score = torch.sigmoid(x)
            else:
                cls_score = torch.softmax(x, -1)

            grad_cams = self.get_gradcam(cls_score)
            return cls_score, cam, grad_cams
        return x


def resnet18(pretrained=False, input_shapes=(3, 224, 244), **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_shapes=input_shapes, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, input_shapes=(3, 224, 244), **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_shapes=input_shapes, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, input_shapes=(3, 224, 244), **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_shapes=input_shapes, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, input_shapes=(3, 224, 244), **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_shapes=input_shapes, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, input_shapes=(3, 224, 244), **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], input_shapes=input_shapes, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


