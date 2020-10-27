import torch
import torch.nn as nn

import resnet
import torch.nn.functional as F
import mmd

class Classifer(nn.Module):
    def __init__(self, num_classes, fc_dim, bottle_neck=False):
        super(Classifer, self).__init__()

        # add classifier and lmmd
        self.cls_num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottle_neck = bottle_neck
        in_features = fc_dim
        if self.bottle_neck:
            self.bottle = nn.Linear(in_features, 256)
            self.fc = nn.Linear(256, self.cls_num_classes)
        else:
            self.fc = nn.Linear(in_features, self.cls_num_classes)

    def forward(self, x, target=None, s_label=None, t_label=None):
        batch_size = x.shape[0]

        src_feat = self.avgpool(x).view(batch_size, -1) # 
        if self.bottle_neck:
            src_feat = self.bottle(src_feat)
        src_pred = self.fc(src_feat)

        if target != None and s_label != None and self.training ==True:
            tgt_feat = self.avgpool(target).view(batch_size, -1) #

            if self.bottle_neck:
                tgt_feat = self.bottle(tgt_feat)
            tgt_pred = self.fc(tgt_feat)
            
            if t_label == None: 
                loss_mmd = mmd.lmmd(src_feat, tgt_feat, s_label, torch.nn.functional.softmax(tgt_pred, dim=1))
                return src_pred, loss_mmd, None
            else:
                loss_mmd = mmd.lmmd(src_feat, tgt_feat, s_label, t_label)
                return src_pred, loss_mmd, tgt_pred

        elif target != None and self.training ==True:
            tgt_feat = self.avgpool(target).view(batch_size, -1) #
            if self.bottle_neck:
                tgt_feat = self.bottle(tgt_feat)
            tgt_pred = self.fc(tgt_feat)

            return src_pred, tgt_pred
            

        return src_pred


class SegClsModule(nn.Module):
    def __init__(self, args):
        super(SegClsModule, self).__init__()
        self.args = args
        builder = ModelBuilder()
        self.encoder = builder.build_encoder(arch=args.encoder_arch, weights=args.encoder_weights, fc_dim=args.fc_dim)
        # print(self.encoder.last_conv_channels)
        # assert 0
        if self.args.do_seg:
            self.decoder = builder.build_decoder(arch=args.decoder_arch, fc_dim=args.fc_dim, num_class=args.seg_num_classes, weights=args.decoder_weights, use_aux=args.use_aux)
        if self.args.do_cls:
            self.classifier = builder.build_clssifier(args.cls_num_classes, fc_dim=args.fc_dim, bottle_neck=args.bottle_neck)
        
        self.use_aux = args.use_aux
        


    def forward(self, src_img, tgt_img=None, c_label=None, s_label=None, t_label=None):
        interp_size = src_img.size()[2:]
        src_conv_out = self.encoder(src_img)

        if tgt_img != None:
            #assert tgt_img == None, "tgt_img is None"
            tgt_conv_out = self.encoder(tgt_img)


        if self.args.do_cls:
            if self.args.do_cls_mmd and c_label!=None and tgt_img != None:
                #assert c_label == None, "c_label in cls_mmd is None"
                src_cls_logits, cls_lmmd_loss, tgt_cls_logits = self.classifier(src_conv_out[-1], tgt_conv_out[-1], c_label, t_label=t_label)
            elif tgt_img != None:
                src_cls_logits, tgt_cls_logits = self.classifier(src_conv_out[-1], tgt_conv_out[-1])
                cls_lmmd_loss = None
            else:
                src_cls_logits = self.classifier(src_conv_out[-1])
                cls_lmmd_loss = None
                tgt_cls_logits = None
        else:
            src_cls_logits = None
            cls_lmmd_loss = None
            tgt_cls_logits = None
            
        if self.args.do_seg:
            if self.args.do_seg_mmd and s_label != None and tgt_img != None:
                #assert s_label == None, "s_label in seg_mmd is None"
                src_seg_logits, seg_lmmd_loss = self.decoder(src_conv_out, tgt_conv_out, s_label)

            else:
                src_seg_logits = self.decoder(src_conv_out)
                seg_lmmd_loss = None

            src_seg_logits = nn.functional.upsample(src_seg_logits, interp_size, mode='bilinear', align_corners=False)
        else:
            src_seg_logits = None
            seg_lmmd_loss = None  
        
        return src_cls_logits, cls_lmmd_loss, src_seg_logits, seg_lmmd_loss, tgt_cls_logits


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        # elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        pretrained = True
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18_dilated8':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet18_dilated16':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50_dilated8':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet50_dilated16':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101_dilated8':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet101_dilated16':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        else:
            raise Exception('Architecture undefined!')

        if weights is not None:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_clssifier(self, num_classes=2, fc_dim=512, bottle_neck=False):
    
        classifier = Classifer(num_classes, fc_dim, bottle_neck=False)
        classifier.apply(self.weights_init)
        return classifier


    def build_decoder(self, arch='ppm_bilinear_deepsup',
                      fc_dim=512, num_class=3,
                      weights='', use_aux=True):
        if arch == 'fcn':
            net_decoder = FCN(
                num_class=num_class,
                fc_dim=fc_dim,
                use_aux=use_aux
            )
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if weights is not None:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.last_conv_channels = orig_resnet.fc.in_features

    def forward(self, x):
        # conv_out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv3 = x
        x = self.layer4(x)

        return [x_conv3, x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv3 = x
        x = self.layer4(x)

        # if return_feature_maps:
            # return conv_out
        return [x_conv3, x]




class FCN(nn.Module):
    def __init__(self, num_class=19, fc_dim=4096, use_aux=False):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(fc_dim, 512, 3, padding=12, dilation=12)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, num_class, 3, padding=12, dilation=12)

    def forward(self, x, target=None, s_label=None):
        x = x[-1]
        x = self.relu1(self.conv1(x)) # 

        if target is not None and s_label is not None:
            target = target[-1]
            target = self.relu1(self.conv1(target))

            t_label = torch.softmax(self.conv2(target), dim=1)
            t_label = t_label.view(t_label.size(0), t_label.size(1), -1)
            
            
            target = target.view(target.size(0), target.size(1), -1)
            
            s_label = s_label.view(s_label.size(0), s_label.size(1), -1)

            source = x.view(x.size(0), x.size(1), -1)

            seg_lmmd_loss = 0
            num = source.size(2)
            for idx in range(num):
                seg_lmmd_loss += mmd.lmmd(source[...,idx], target[...,idx], s_label[...,0,idx], t_label[...,idx])
            
            seg_lmmd_loss = seg_lmmd_loss / num

            return self.conv2(x), seg_lmmd_loss

        return self.conv2(x)


