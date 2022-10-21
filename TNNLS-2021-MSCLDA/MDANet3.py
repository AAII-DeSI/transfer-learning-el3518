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


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256*6*6)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)
        return x

class MDAnet(nn.Module):

    def __init__(self, num_classes=31):
        super(MDAnet, self).__init__()
        
        self.sharedNet = resnet50(True)
        
        self.sonnetc1 = ADDneck(2048, 256)
        self.sonnets1 = ADDneck(2048, 256)
        
        self.sonnetc2 = ADDneck(2048, 256)
        self.sonnets2 = ADDneck(2048, 256)
        
        self.sonnetc3 = ADDneck(2048, 256)
        self.sonnets3 = ADDneck(2048, 256)

        self.cls_fc_son1 = nn.Linear(512, num_classes)
        self.cls_fc_son2 = nn.Linear(512, num_classes)
        self.cls_fc_son3 = nn.Linear(512, num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))##alexnet
        self.classes = num_classes

    def forward(self, data_src1, data_src2=0, data_src3=0, data_tgt = 0, label_src = 0, mark = 1):
        com_loss = 0
        div_loss = 0
        st_loss = 0
        
        intra_loss = 0
        inters_loss = 0
        intert_loss = 0
        
        st_lossp = 0
        intra_lossp = 0
        
        
               
        if self.training == True:
            data_src1 = self.sharedNet(data_src1)
            data_src2 = self.sharedNet(data_src2) 
            data_src3 = self.sharedNet(data_src3)       
            data_tgt = self.sharedNet(data_tgt)
            
            data_src1p = self.avgpool(data_src1)
            data_src2p = self.avgpool(data_src2)
            data_src3p = self.avgpool(data_src3)
            data_tgt_p = self.avgpool(data_tgt)
        
            data_src1p = data_src1p.view(data_src1p.size(0), -1)
            data_src2p = data_src2p.view(data_src2p.size(0), -1)
            data_src3p = data_src3p.view(data_src3p.size(0), -1)
            data_tgt_p = data_tgt_p.view(data_tgt_p.size(0), -1) 

            data_tgtc1 = self.sonnetc1(data_tgt)
            data_tgts1 = self.sonnets1(data_tgt)
            data_tgtc1 = self.avgpool(data_tgtc1)
            data_tgtc1 = data_tgtc1.view(data_tgtc1.size(0), -1)
            data_tgts1 = self.avgpool(data_tgts1)
            data_tgts1 = data_tgts1.view(data_tgts1.size(0), -1)
            
            data_tgtc2 = self.sonnetc2(data_tgt)
            data_tgts2 = self.sonnets2(data_tgt)
            data_tgtc2 = self.avgpool(data_tgtc2)
            data_tgtc2 = data_tgtc2.view(data_tgtc2.size(0), -1)
            data_tgts2 = self.avgpool(data_tgts2)
            data_tgts2 = data_tgts2.view(data_tgts2.size(0), -1)
            
            data_tgtc3 = self.sonnetc3(data_tgt)
            data_tgts3 = self.sonnets3(data_tgt)
            data_tgtc3 = self.avgpool(data_tgtc3)
            data_tgtc3 = data_tgtc3.view(data_tgtc3.size(0), -1)
            data_tgts3 = self.avgpool(data_tgts3)
            data_tgts3 = data_tgts3.view(data_tgts3.size(0), -1)           
            
            data_tgt_1 = torch.cat((data_tgtc1, data_tgts1), 1)
            pred_tgt1 = self.cls_fc_son1(data_tgt_1)
            del data_tgtc1
            del data_tgts1
                                   
            data_tgt_2 = torch.cat((data_tgtc2, data_tgts2), 1)
            pred_tgt2 = self.cls_fc_son2(data_tgt_2)
            del data_tgtc2
            del data_tgts2
            
            data_tgt_3 = torch.cat((data_tgtc3, data_tgts3), 1)
            pred_tgt3 = self.cls_fc_son3(data_tgt_3)
            del data_tgtc3
            del data_tgts3
            
            data_srcc1 = self.sonnetc1(data_src1)
            data_srcs1 = self.sonnets1(data_src1)
            data_srcc1 = self.avgpool(data_srcc1)
            data_srcc1 = data_srcc1.view(data_srcc1.size(0), -1)
            data_srcs1 = self.avgpool(data_srcs1)
            data_srcs1 = data_srcs1.view(data_srcs1.size(0), -1)
            
            data_srcc2 = self.sonnetc2(data_src2)
            data_srcs2 = self.sonnets2(data_src2)
            data_srcc2 = self.avgpool(data_srcc2)
            data_srcc2 = data_srcc2.view(data_srcc2.size(0), -1)
            data_srcs2 = self.avgpool(data_srcs2)
            data_srcs2 = data_srcs2.view(data_srcs2.size(0), -1)
            
            data_srcc3 = self.sonnetc3(data_src3)
            data_srcc3 = self.avgpool(data_srcc3)
            data_srcc3 = data_srcc3.view(data_srcc3.size(0), -1)
            data_srcs3 = self.sonnets3(data_src3)
            data_srcs3 = self.avgpool(data_srcs3)
            data_srcs3 = data_srcs3.view(data_srcs3.size(0), -1)
            data_src3 = torch.cat((data_srcc3, data_srcs3), 1)

            com_loss += mmd.mmd(data_srcc1, data_srcc2) 
            com_loss += mmd.mmd(data_srcc1, data_srcc3)
            com_loss += mmd.mmd(data_srcc2, data_srcc3)
            
            div_loss += mmd.mmd(data_srcs1, data_srcs2) #max   
            div_loss += mmd.mmd(data_srcs1, data_srcs3)
            div_loss += mmd.mmd(data_srcs2, data_srcs3)        

            if mark == 1:
                
                st_lossp += mmd.mmd(data_src1p, data_tgt_p)               
                
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt2, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)) )

                data_src = torch.cat((data_srcc1, data_srcs1), 1)
                del data_srcc1
                del data_srcs1
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                st_loss += mmd.mmd(data_src, data_tgt_1)
                s_label = self.softmax(pred_src)
                t_label = self.softmax(pred_tgt1)
                del pred_src

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
                    intra_loss += mmd.mmd(ps * data_src, pt * data_tgt_1)
                    intra_lossp += mmd.mmd(ps * data_src1p, pt * data_tgt_p)
                
                ps1 = s_label[:, smax].reshape(data_src.shape[0],1)
                ps2 = s_label[:, smax2].reshape(data_src.shape[0],1)
                inters_loss += mmd.mmd(ps1 * data_src, ps2 * data_src)

                pt1 = t_label[:, tmax].reshape(data_src.shape[0],1)
                pt2 = t_label[:, tmax2].reshape(data_src.shape[0],1)
                intert_loss += mmd.mmd(pt1 * data_tgt_1, pt2 * data_tgt_1)

                domain_loss = st_loss + (com_loss  - div_loss)/3
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                weights = st_lossp + intra_lossp / self.classes 
                
                return domain_loss, class_loss, cls_loss, l1_loss/3, weights#, s_loss

            if mark == 2:

                st_lossp += mmd.mmd(data_src2p, data_tgt_p)

                
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt3, dim=1)) )
                
                data_src = torch.cat((data_srcc2, data_srcs2), 1)
                del data_srcc2
                del data_srcs2
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                st_loss += mmd.mmd(data_src, data_tgt_2)
                s_label = self.softmax(pred_src)
                t_label = self.softmax(pred_tgt2)
                del pred_src

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
                    intra_loss += mmd.mmd(ps * data_src, pt * data_tgt_2)
                    intra_lossp += mmd.mmd(ps * data_src2p, pt * data_tgt_p)
                
                ps1 = s_label[:, smax].reshape(data_src.shape[0],1)
                ps2 = s_label[:, smax2].reshape(data_src.shape[0],1)
                inters_loss += mmd.mmd(ps1 * data_src, ps2 * data_src)

                pt1 = t_label[:, tmax].reshape(data_src.shape[0],1)
                pt2 = t_label[:, tmax2].reshape(data_src.shape[0],1)
                intert_loss += mmd.mmd(pt1 * data_tgt_2, pt2 * data_tgt_2)

                domain_loss = st_loss + (com_loss - div_loss)/3
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                weights = st_lossp + intra_lossp / self.classes 
                
                return domain_loss, class_loss, cls_loss, l1_loss/3, weights#, s_loss

            if mark == 3:

                st_lossp += mmd.mmd(data_src3p, data_tgt_p)
                                              
                
                l1_loss = torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt1, dim=1)) )
                l1_loss += torch.mean( torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt2, dim=1)) )
                
                data_src = torch.cat((data_srcc3, data_srcs3), 1)
                del data_srcc3
                del data_srcs3
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                st_loss += mmd.mmd(data_src, data_tgt_3)
                s_label = self.softmax(pred_src)
                t_label = self.softmax(pred_tgt3)
                del pred_src
                del pred_tgt1
                del pred_tgt2
                del pred_tgt3
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
                    intra_loss += mmd.mmd(ps * data_src, pt * data_tgt_3)
                    intra_lossp += mmd.mmd(ps * data_src3p, pt * data_tgt_p)
                
                ps1 = s_label[:, smax].reshape(data_src.shape[0],1)
                ps2 = s_label[:, smax2].reshape(data_src.shape[0],1)
                inters_loss += mmd.mmd(ps1 * data_src, ps2 * data_src)

                pt1 = t_label[:, tmax].reshape(data_src.shape[0],1)
                pt2 = t_label[:, tmax2].reshape(data_src.shape[0],1)
                intert_loss += mmd.mmd(pt1 * data_tgt_3, pt2 * data_tgt_3)

                domain_loss = st_loss + (com_loss - div_loss)/3
                class_loss =  intra_loss /  self.classes - 0.01*(inters_loss + intert_loss)/2
                weights = st_lossp + intra_lossp / self.classes 
                
                return domain_loss, class_loss, cls_loss, l1_loss/3, weights#, s_loss

        else:
            
            data = self.sharedNet(data_src1)

            feac1 = self.sonnetc1(data)
            feas1 = self.sonnets1(data)
            feac1 = self.avgpool(feac1)
            feac1 = feac1.view(feac1.size(0), -1)
            feas1 = self.avgpool(feas1)
            feas1 = feas1.view(feas1.size(0), -1)

            fea1 = torch.cat((feac1, feas1), 1)
            pred1 = self.cls_fc_son1(fea1)

            feac2 = self.sonnetc2(data)
            feas2 = self.sonnets2(data)
            feac2 = self.avgpool(feac2)
            feac2 = feac2.view(feac2.size(0), -1)
            feas2 = self.avgpool(feas2)
            feas2 = feas2.view(feas2.size(0), -1)
            
            fea2 = torch.cat((feac2, feas2), 1)
            pred2 = self.cls_fc_son2(fea2)
            
            fea_sonc3 = self.sonnetc3(data)
            fea_sons3 = self.sonnets3(data)
            fea_sonc3 = self.avgpool(fea_sonc3)
            fea_sonc3 = fea_sonc3.view(fea_sonc3.size(0), -1)           
            fea_sons3 = self.avgpool(fea_sons3)
            fea_sons3 = fea_sons3.view(fea_sons3.size(0), -1)
            
            fea_son3 = torch.cat((fea_sonc3, fea_sons3), 1)
            pred3 = self.cls_fc_son3(fea_son3)
            
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
