import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
#from torch.autograd import Variable
import torch
import numpy as np


__all__ = ['ResNet', 'resnet50', 'resnet101', 'AlexNet', 'alexnet']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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
        super(Bottleneck, self).__init__()
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

class ADDneck(nn.Module):
#inplanes=2048 planes=256
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
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
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

        return x

def avgp_fea(self, out):
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    return out

def cal_cls_loss(self, data_src, data_tgt, pred_src, pred_tgt):
    s_label = self.softmax(pred_src)
    t_label = self.softmax(pred_tgt)

    sums_label = s_label.data.sum(0)
    sumt_label = t_label.data.sum(0)
    smax = sums_label.data.max(0)[1]
    tmax = sumt_label.data.max(0)[1]
    sums_label[smax] = 0
    sumt_label[tmax] = 0

    smax2 = sums_label.data.max(0)[1]
    tmax2 = sumt_label.data.max(0)[1]

    for c in range(self.classes):
        ps = s_label[:, c].reshape(data_src.shape[0],1)
        pt = t_label[:, c].reshape(data_src.shape[0],1)
        intra_loss = mmd.mmd(ps * data_src, pt * data_tgt)
                
    ps1 = s_label[:, smax].reshape(data_src.shape[0],1)
    ps2 = s_label[:, smax2].reshape(data_src.shape[0],1)
    inters_loss = mmd.mmd(ps1 * data_src, ps2 * data_src)

    pt1 = t_label[:, tmax].reshape(data_src.shape[0],1)
    pt2 = t_label[:, tmax2].reshape(data_src.shape[0],1)
    intert_loss = mmd.mmd(pt1 * data_tgt, pt2 * data_tgt)
    
    return intra_loss, inters_loss, intert_loss

class MDAnetsf(nn.Module):

    def __init__(self, num_classes=31):
        super(MDAnetsf, self).__init__()
        
        #self.sharedNet = resnet50(True)
        self.sharedNet = resnet101(True)
        
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)        
        self.sonnet3 = ADDneck(2048, 256)

        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))##alexnet
        self.classes = num_classes

    def forward(self, data_src, label_src = 0, mark = 1):
            
        if self.training == True:
            data_src = self.sharedNet(data_src)       

            if mark == 1:              
                
                data_src = self.sonnet1(data_src)

                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son2(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son3(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                return cls_loss

            if mark == 2:

                data_src = self.sonnet2(data_src)
                
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son1(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son3(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                return cls_loss

            if mark == 3:

                data_src = self.sonnet3(data_src)

                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son1(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                pred_src = self.cls_fc_son2(data_src)
                cls_loss += 0.5*F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                                
                return cls_loss

        else:
            
            data = self.sharedNet(data_src)

            fea1 = self.sonnet1(data)
            pred1 = self.cls_fc_son1(fea1)

            fea2 = self.sonnet2(data)
            pred2 = self.cls_fc_son2(fea2)
            
            fea3 = self.sonnet3(data)
            pred3 = self.cls_fc_son3(fea3)
            
            return pred1, pred2, pred3

class MDAdomnet(nn.Module):

    def __init__(self, num_classes=31):
        super(MDAdomnet, self).__init__()        
        
        #self.sharedNet = resnet50(True)
        self.sharedNet = resnet101(True)
        
        self.domain = nn.Sequential()
        self.domain.add_module('fc1', nn.Linear(2048, 256))
        self.domain.add_module('relu1', nn.ReLU(True))
        self.domain.add_module('dpt1', nn.Dropout())
        self.domain.add_module('fc2', nn.Linear(256, 1024))
        self.domain.add_module('relu2', nn.ReLU(True))
        self.domain.add_module('dpt2', nn.Dropout())
        self.domain.add_module('fc3', nn.Linear(1024, 3))

        #self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.classes = num_classes

    def forward(self, data_src1, data_src2 = 0, data_src3 = 0):

        if self.training == True:
            
            
            data_src1 = avgp_fea(self, self.sharedNet(data_src1))
            data_src2 = avgp_fea(self, self.sharedNet(data_src2))
            data_src3 = avgp_fea(self, self.sharedNet(data_src3))
            
            
            pred_domain = self.domain(data_src1)
            s_label = torch.zeros(data_src1.shape[0]).long().cuda()
            cls_loss1 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            pred_domain = self.domain(data_src2)
            s_label = torch.ones(data_src1.shape[0]).long().cuda()
            cls_loss2 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            pred_domain = self.domain(data_src3)
            s_label = 2*torch.ones(data_src1.shape[0]).long().cuda()
            cls_loss3 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            cls_loss = cls_loss1 + cls_loss2 + cls_loss3
                                             
            return cls_loss

        else:
            data_src1 = avgp_fea(self, self.sharedNet(data_src1))                        
            pred = self.domain(data_src1)

            return pred

class Discriminator(nn.Module):
    def __init__(self, n=10):
        super(Discriminator, self).__init__()
        
        self.sharedNet = resnet50(True)
        #self.sharedNet = resnet101(True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.n = n
        self.fcs = nn.Sequential()
        self.fc = {}
        for i in range(n):
            self.fc[i] = nn.Sequential(
                         nn.Linear(2048, 256),
                         nn.BatchNorm1d(256),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Linear(256, 1),
                         nn.Sigmoid()
                         )
            self.fcs.add_module('fc_'+str(i), self.fc[i])
        '''
        self.n = n
        def f():
            return nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d'%i, f())
        '''
    
    def forward(self, x):
        x = avgp_fea(self, self.sharedNet(x))
        outs = [self.fcs[i](x) for i in range(self.n)]
        #outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class TSSnet(nn.Module):

    def __init__(self, num_classes=31):
        super(TSSnet, self).__init__()
        
        #self.sharedNet = resnet50(True)
        self.sharedNet = resnet101(True) 
        
        self.sonnet1 = ADDneck(2048, 256)
        self.sonnet2 = ADDneck(2048, 256)        
        self.sonnet3 = ADDneck(2048, 256)


        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        
        
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))##alexnet
        self.classes = num_classes

    def forward(self, data_src, data_srct=0, data_tst=0, 
                data_tgt = 0, label_src = 0, label_srct = 0, mark = 1):

        tar_loss = 0
        st_loss = 0
        
        intra_loss = 0
        inters_loss = 0
        intert_loss = 0                        
               
        if self.training == True:
            data_src = self.sharedNet(data_src)
                        
            data_srct = self.sharedNet(data_srct)
            data_tst = self.sharedNet(data_tst) 
            data_tgt = self.sharedNet(data_tgt) 
            
            data_tgt1 = self.sonnet1(data_tgt)
            pred_tgt1 = self.cls_fc_son1(data_tgt1)
            
            data_tgt2 = self.sonnet2(data_tgt)
            pred_tgt2 = self.cls_fc_son2(data_tgt2)
            
            data_tgt3 = self.sonnet3(data_tgt)
            pred_tgt3 = self.cls_fc_son3(data_tgt3)
            
            data_srct1 = self.sonnet1(data_srct)           
            data_srct2 = self.sonnet2(data_srct)           
            data_srct3 = self.sonnet3(data_srct)             

            if mark == 1:                
                                
                #data_src = self.sharedNet(data_src1)
                data_src = self.sonnet1(data_src)
                data_tst1 = self.sonnet1(data_tst)
                
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt2, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)) )
                
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                #'''
                pred_srct = self.cls_fc_son1(data_srct1)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son2(data_srct2)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son3(data_srct3)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                #'''
                
                tar_loss += mmd.mmd(data_srct1, data_tst1)
                st_loss += mmd.mmd(data_src, data_tgt1)
                
                intra_loss0, inters_loss0, intert_loss0 = cal_cls_loss(self, data_src, data_tgt1, pred_src, pred_tgt1)                    
                intra_loss += intra_loss0
                inters_loss += inters_loss0
                intert_loss += intert_loss0

                domain_loss = st_loss + tar_loss #
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                
                return domain_loss, class_loss, cls_loss, l1_loss/2

            if mark == 2:                                             
                
                data_src = self.sonnet2(data_src)
                data_tst2 = self.sonnet2(data_tst)
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)) )
                
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                #'''
                pred_srct = self.cls_fc_son1(data_srct1)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son2(data_srct2)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son3(data_srct3)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                #'''
                
                tar_loss += mmd.mmd(data_srct2, data_tst2)
                st_loss += mmd.mmd(data_src, data_tgt2)
                
                intra_loss0, inters_loss0, intert_loss0 = cal_cls_loss(self, data_src, data_tgt2, pred_src, pred_tgt2)                    
                intra_loss += intra_loss0
                inters_loss += inters_loss0
                intert_loss += intert_loss0


                domain_loss = st_loss + tar_loss  # 
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                
                return domain_loss, class_loss, cls_loss, l1_loss/2

            if mark == 3:

                data_src = self.sonnet3(data_src)
                data_tst3 = self.sonnet3(data_tst)
                                                              
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt2, dim=1)) )
                
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                #'''
                pred_srct = self.cls_fc_son1(data_srct1)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son2(data_srct2)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                pred_srct = self.cls_fc_son3(data_srct3)                
                cls_loss += F.nll_loss(F.log_softmax(pred_srct, dim=1), label_srct)
                #'''
                
                tar_loss += mmd.mmd(data_srct3, data_tst3)
                st_loss += mmd.mmd(data_src, data_tgt3)
                
                intra_loss0, inters_loss0, intert_loss0 = cal_cls_loss(self, data_src, data_tgt3, pred_src, pred_tgt3)                    
                intra_loss += intra_loss0
                inters_loss += inters_loss0
                intert_loss += intert_loss0

                domain_loss = st_loss + tar_loss #
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                
                return domain_loss, class_loss, cls_loss, l1_loss/2

        
        else:
            
            data = self.sharedNet(data_src)

            fea1 = self.sonnet1(data)
            pred1 = self.cls_fc_son1(fea1)

            fea2 = self.sonnet2(data)
            pred2 = self.cls_fc_son2(fea2)
            
            fea3 = self.sonnet3(data)
            pred3 = self.cls_fc_son3(fea3)
            
            return pred1, pred2, pred3

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
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
