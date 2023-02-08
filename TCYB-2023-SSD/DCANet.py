import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
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

class ADDneck1(nn.Module):
#inplanes=2048 planes=256
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               #padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
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

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)   
     
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(7, stride=1)


    def forward(self, source):
        s_branch1x1 = self.branch1x1(source)

        s_branch5x5 = self.branch5x5_1(source)
        s_branch5x5 = self.branch5x5_2(s_branch5x5)

        s_branch3x3dbl = self.branch3x3dbl_1(source)
        s_branch3x3dbl = self.branch3x3dbl_2(s_branch3x3dbl)
        s_branch3x3dbl = self.branch3x3dbl_3(s_branch3x3dbl)

        s_branch_pool = F.avg_pool2d(source, kernel_size=3, stride=1, padding=1)
        s_branch_pool = self.branch_pool(s_branch_pool)

        s_branch1x1 = self.avg_pool(s_branch1x1)
        s_branch5x5 = self.avg_pool(s_branch5x5)
        s_branch3x3dbl = self.avg_pool(s_branch3x3dbl)
        s_branch_pool = self.avg_pool(s_branch_pool)

        s_branch1x1 = s_branch1x1.view(s_branch1x1.size(0), -1)
        s_branch5x5 = s_branch5x5.view(s_branch5x5.size(0), -1)
        s_branch3x3dbl = s_branch3x3dbl.view(s_branch3x3dbl.size(0), -1)
        s_branch_pool = s_branch_pool.view(s_branch_pool.size(0), -1)       

        out = torch.cat([s_branch1x1, s_branch5x5, s_branch3x3dbl, s_branch_pool], 1)                
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

class DCAnet3(nn.Module):

    def __init__(self, num_classes=31):
        super(DCAnet3, self).__init__()
        
        self.sharedNet = resnet50(True)                
        
        self.sonnetc1 = ADDneck1(2048, 256)
        self.sonnets1 = ADDneck(2048, 256)
        self.sonnet1 = InceptionA(2048, 64)
        
        self.sonnetc2 = ADDneck1(2048, 256)
        self.sonnets2 = ADDneck(2048, 256)
        self.sonnet2 = InceptionA(2048, 64)
        
        self.sonnetc3 = ADDneck1(2048, 256)
        self.sonnets3 = ADDneck(2048, 256)
        self.sonnet3 = InceptionA(2048, 64)

        self.cls_fc_son11 = nn.Linear(256, num_classes)
        self.cls_fc_son21 = nn.Linear(256, num_classes) 
        self.cls_fc_son31 = nn.Linear(256, num_classes)
        
        self.cls_fc_son12 = nn.Linear(256, num_classes)
        self.cls_fc_son22 = nn.Linear(256, num_classes)
        self.cls_fc_son32 = nn.Linear(256, num_classes)
        
        self.cls_fc_son13 = nn.Linear(288, num_classes)
        self.cls_fc_son23 = nn.Linear(288, num_classes) 
        self.cls_fc_son33 = nn.Linear(288, num_classes) 
        
        self.member1 = nn.Linear(3, 1)
        self.mweight1 = nn.Sequential()
        self.mweight1.add_module('fc1', nn.Linear(num_classes, 256))
        self.mweight1.add_module('relu1', nn.ReLU(True))
        self.mweight1.add_module('dpt1', nn.Dropout())       
        self.mweight1.add_module('fc2', nn.Linear(256, 3))
        
        self.member2 = nn.Linear(3, 1)
        self.mweight2 = nn.Sequential()
        self.mweight2.add_module('fc1', nn.Linear(num_classes, 256))
        self.mweight2.add_module('relu1', nn.ReLU(True))
        self.mweight2.add_module('dpt1', nn.Dropout())       
        self.mweight2.add_module('fc2', nn.Linear(256, 3))
        
        self.member3 = nn.Linear(3, 1)
        self.mweight3 = nn.Sequential()
        self.mweight3.add_module('fc1', nn.Linear(num_classes, 256))
        self.mweight3.add_module('relu1', nn.ReLU(True))
        self.mweight3.add_module('dpt1', nn.Dropout())       
        self.mweight3.add_module('fc2', nn.Linear(256, 3))
                        
        self.domain = nn.Sequential()
        self.domain.add_module('fc1', nn.Linear(2048, 256))
        self.domain.add_module('relu1', nn.ReLU(True))
        self.domain.add_module('dpt1', nn.Dropout())
        self.domain.add_module('fc2', nn.Linear(256, 1024))
        self.domain.add_module('relu2', nn.ReLU(True))
        self.domain.add_module('dpt2', nn.Dropout())
        self.domain.add_module('fc3', nn.Linear(1024, 3))                             
        
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classes = num_classes

    def forward(self, data_src1, data_src2=0, data_src3=0, data_srct=0, data_tgt = 0, label_src = 0, label_srct = 0, epo = 0, pse_num = 0, mark = 1):
        st_loss = 0 
        cls_losst = 0  
        dis_loss = 0                                 
               
        if self.training == True:

            data_src1 = self.sharedNet(data_src1)
            data_src2 = self.sharedNet(data_src2)   
            data_src3 = self.sharedNet(data_src3)     
            data_tgt = self.sharedNet(data_tgt)  
            
            data_src1p = self.avgpool(data_src1)
            data_src2p = self.avgpool(data_src2)
            data_src3p = self.avgpool(data_src3)
        
            data_src1p = data_src1p.view(data_src1p.size(0), -1)
            data_src2p = data_src2p.view(data_src2p.size(0), -1)
            data_src3p = data_src3p.view(data_src3p.size(0), -1)
            
            pred_domain = self.domain(data_src1p)
            s_label = torch.zeros(data_src1.shape[0]).long().cuda()
            cls_loss1 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            pred_domain = self.domain(data_src2p)
            s_label = torch.ones(data_src1.shape[0]).long().cuda()
            cls_loss2 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            pred_domain = self.domain(data_src3p)
            s_label = 2*torch.ones(data_src1.shape[0]).long().cuda()
            cls_loss3 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
            domain_loss = cls_loss1 + cls_loss2 + cls_loss3          

            data_tgtc1 = self.sonnetc1(data_tgt)
            data_tgts1 = self.sonnets1(data_tgt)
            data_tgt1 = self.sonnet1(data_tgt)
            data_tgtc1 = self.avgpool(data_tgtc1)
            data_tgtc1 = data_tgtc1.view(data_tgtc1.size(0), -1)
            data_tgts1 = self.avgpool(data_tgts1)
            data_tgts1 = data_tgts1.view(data_tgts1.size(0), -1)
            
            data_tgtc2 = self.sonnetc2(data_tgt)
            data_tgts2 = self.sonnets2(data_tgt)
            data_tgt2 = self.sonnet2(data_tgt)
            data_tgtc2 = self.avgpool(data_tgtc2)
            data_tgtc2 = data_tgtc2.view(data_tgtc2.size(0), -1)
            data_tgts2 = self.avgpool(data_tgts2)
            data_tgts2 = data_tgts2.view(data_tgts2.size(0), -1)
            
            data_tgtc3 = self.sonnetc3(data_tgt)
            data_tgts3 = self.sonnets3(data_tgt)
            data_tgt3 = self.sonnet3(data_tgt)
            data_tgtc3 = self.avgpool(data_tgtc3)
            data_tgtc3 = data_tgtc3.view(data_tgtc3.size(0), -1)
            data_tgts3 = self.avgpool(data_tgts3)
            data_tgts3 = data_tgts3.view(data_tgts3.size(0), -1)

            pred_tgt11 = self.cls_fc_son11(data_tgtc1)
            pred_tgt12 = self.cls_fc_son12(data_tgts1)
            pred_tgt13 = self.cls_fc_son13(data_tgt1)
            pred_data = torch.stack([pred_tgt11, pred_tgt12, pred_tgt13], 2)
            pred_tgt1 = self.member1(pred_data)
            pred_tgt1 = torch.squeeze(pred_tgt1,2)
            
            
            weight =  self.softmax(self.mweight1(pred_tgt1)) 
            r11=weight[:, 0].reshape(data_tgt1.shape[0],1) 
            r12=weight[:, 1].reshape(data_tgt1.shape[0],1) 
            r13=weight[:, 2].reshape(data_tgt1.shape[0],1) 
            pred_tgt1 = r11*pred_tgt11 + r12*pred_tgt12 + r13*pred_tgt13
            

            pred_tgt21 = self.cls_fc_son21(data_tgtc2)
            pred_tgt22 = self.cls_fc_son22(data_tgts2)
            pred_tgt23 = self.cls_fc_son23(data_tgt2)
            pred_data = torch.stack([pred_tgt21, pred_tgt22, pred_tgt23], 2)
            pred_tgt2 = self.member2(pred_data)
            pred_tgt2 = torch.squeeze(pred_tgt2,2)  
            
            weight =  self.softmax(self.mweight2(pred_tgt2)) 
            r21=weight[:, 0].reshape(data_tgt1.shape[0],1) 
            r22=weight[:, 1].reshape(data_tgt1.shape[0],1) 
            r23=weight[:, 2].reshape(data_tgt1.shape[0],1)
            pred_tgt2 = r21*pred_tgt21 + r22*pred_tgt22 + r23*pred_tgt23
             
            
            pred_tgt31 = self.cls_fc_son31(data_tgtc3)
            pred_tgt32 = self.cls_fc_son32(data_tgts3)
            pred_tgt33 = self.cls_fc_son33(data_tgt3)
            pred_data = torch.stack([pred_tgt31, pred_tgt32, pred_tgt33], 2)
            pred_tgt3 = self.member3(pred_data)
            pred_tgt3 = torch.squeeze(pred_tgt3,2)
            
            weight =  self.softmax(self.mweight3(pred_tgt3)) 
            r31=weight[:, 0].reshape(data_tgt1.shape[0],1) 
            r32=weight[:, 1].reshape(data_tgt1.shape[0],1) 
            r33=weight[:, 2].reshape(data_tgt1.shape[0],1)
            pred_tgt3 = r31*pred_tgt31 + r32*pred_tgt32 + r33*pred_tgt33 
                                             
                                   
            if mark == 1:                
                data_srcc1 = self.sonnetc1(data_src1)
                data_srcs1 = self.sonnets1(data_src1)
                data_src1 = self.sonnet1(data_src1)
                data_srcc1 = self.avgpool(data_srcc1)
                data_srcc1 = data_srcc1.view(data_srcc1.size(0), -1)
                data_srcs1 = self.avgpool(data_srcs1)
                data_srcs1 = data_srcs1.view(data_srcs1.size(0), -1)                                

                pred_src11 = self.cls_fc_son11(data_srcc1)
                pred_src12 = self.cls_fc_son12(data_srcs1)
                pred_src13 = self.cls_fc_son13(data_src1) 
                pred_src = torch.stack([pred_src11, pred_src12, pred_src13], 2)
                pred_src = self.member1(pred_src)
                pred_src1 = torch.squeeze(pred_src,2)
                
                pred_domain = self.mweight1(pred_src11)
                s_label = torch.zeros(pred_src.shape[0]).long().cuda()
                cls_loss1 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight1(pred_src12)
                s_label = torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss2 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight1(pred_src13)
                s_label = 2*torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss3 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                dis_loss = cls_loss1 + cls_loss2 + cls_loss3
                weight =  self.softmax(self.mweight1(pred_src1)) 
                r11=weight[:, 0].reshape(pred_src.shape[0],1) 
                r12=weight[:, 1].reshape(pred_src.shape[0],1) 
                r13=weight[:, 2].reshape(pred_src.shape[0],1) 
                pred_src = r11*pred_src11 + r12*pred_src12 + r13*pred_src13 
                
                l1_loss = torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1) - torch.nn.functional.softmax(pred_tgt2, dim=1))
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_tgt1, dim=1) - torch.nn.functional.softmax(pred_tgt3, dim=1))
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_src, dim=1) - torch.nn.functional.softmax(pred_src1, dim=1))               
                l1_loss = torch.mean(l1_loss)
                
                cls_loss = F.nll_loss(F.log_softmax(pred_src11, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src12, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src13, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                if epo > pse_num:
                    data_st = self.sharedNet(data_srct)
                    data_tgt_p = self.avgpool(data_st)
                    data_tgt_p = data_tgt_p.view(data_tgt_p.size(0), -1)

                    data_stc1 = self.sonnetc1(data_st)
                    data_sts1 = self.sonnets1(data_st)
                    data_st1 = self.sonnet1(data_st)
                    data_stc1 = self.avgpool(data_stc1)
                    data_stc1 = data_stc1.view(data_stc1.size(0), -1)
                    data_sts1 = self.avgpool(data_sts1)
                    data_sts1 = data_sts1.view(data_sts1.size(0), -1)
                    
                    data_stc2 = self.sonnetc2(data_st)
                    data_sts2 = self.sonnets2(data_st)
                    data_st2 = self.sonnet2(data_st)
                    data_stc2 = self.avgpool(data_stc2)
                    data_stc2 = data_stc2.view(data_stc2.size(0), -1)
                    data_sts2 = self.avgpool(data_sts2)
                    data_sts2 = data_sts2.view(data_sts2.size(0), -1)
                    
                    data_stc3 = self.sonnetc3(data_st)
                    data_sts3 = self.sonnets3(data_st)
                    data_st3 = self.sonnet3(data_st)
                    data_stc3 = self.avgpool(data_stc3)
                    data_stc3 = data_stc3.view(data_stc3.size(0), -1)
                    data_sts3 = self.avgpool(data_sts3)
                    data_sts3 = data_sts3.view(data_sts3.size(0), -1)

                    pred_st11 = self.cls_fc_son11(data_stc1)
                    pred_st12 = self.cls_fc_son12(data_sts1)
                    pred_st13 = self.cls_fc_son13(data_st1)
                    pred_data = torch.stack([pred_st11, pred_st12, pred_st13], 2)
                    pred_st1 = self.member1(pred_data)
                    pred_st1 = torch.squeeze(pred_st1,2)

                    weight =  self.softmax(self.mweight1(pred_st1)) 
                    r11=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r12=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r13=weight[:, 2].reshape(data_st1.shape[0],1) 
                    pred_st1 = r11*pred_st11 + r12*pred_st12 + r13*pred_st13

                    cls_losst = F.nll_loss(F.log_softmax(pred_st11, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st12, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st13, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st1, dim=1), label_srct)

                    pred_st21 = self.cls_fc_son21(data_stc2)
                    pred_st22 = self.cls_fc_son22(data_sts2)
                    pred_st23 = self.cls_fc_son23(data_st2)
                    pred_data = torch.stack([pred_st21, pred_st22, pred_st23], 2)
                    pred_st2 = self.member2(pred_data)
                    pred_st2 = torch.squeeze(pred_st2,2)

                    weight =  self.softmax(self.mweight2(pred_st2)) 
                    r21=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r22=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r23=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st2 = r21*pred_st21 + r22*pred_st22 + r23*pred_st23
                    
                    pred_st31 = self.cls_fc_son31(data_stc3)
                    pred_st32 = self.cls_fc_son32(data_sts3)
                    pred_st33 = self.cls_fc_son33(data_st3)
                    pred_data = torch.stack([pred_st31, pred_st32, pred_st33], 2)
                    pred_st3 = self.member3(pred_data)
                    pred_st3 = torch.squeeze(pred_st3,2)

                    weight =  self.softmax(self.mweight3(pred_st3)) 
                    r31=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r32=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r33=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st3 = r31*pred_st31 + r32*pred_st32 + r33*pred_st33
                    
                    member = self.softmax(self.domain(data_tgt_p))
                    w1=member[:, 0].reshape(pred_st1.shape[0],1) 
                    w2=member[:, 1].reshape(pred_st1.shape[0],1)
                    w3=member[:, 2].reshape(pred_st1.shape[0],1)
                    cls_losst += F.nll_loss(F.log_softmax(w1*pred_st1+w2*pred_st2+w3*pred_st3, dim=1), label_srct)
                                                        
                st_loss += mmd.mmd(data_srcc1, data_tgtc1)
                st_loss += mmd.mmd(data_srcs1, data_tgts1)
                st_loss += mmd.mmd(data_src1, data_tgt1) 
                                                                                                     
                return st_loss, cls_loss + cls_losst, l1_loss/3, domain_loss/3+dis_loss/3

            if mark == 2:

                data_srcc2 = self.sonnetc2(data_src2)
                data_srcs2 = self.sonnets2(data_src2)
                data_src2 = self.sonnet2(data_src2)
                data_srcc2 = self.avgpool(data_srcc2)
                data_srcc2 = data_srcc2.view(data_srcc2.size(0), -1)
                data_srcs2 = self.avgpool(data_srcs2)
                data_srcs2 = data_srcs2.view(data_srcs2.size(0), -1)                                                                              
                
                pred_src21 = self.cls_fc_son21(data_srcc2)
                pred_src22 = self.cls_fc_son22(data_srcs2)
                pred_src23 = self.cls_fc_son23(data_src2)
                pred_src = torch.stack([pred_src21, pred_src22, pred_src23], 2)
                pred_src = self.member2(pred_src)
                pred_src1 = torch.squeeze(pred_src,2)

                pred_domain = self.mweight2(pred_src21)
                s_label = torch.zeros(pred_src.shape[0]).long().cuda()
                cls_loss1 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight2(pred_src22)
                s_label = torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss2 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight2(pred_src23)
                s_label = 2*torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss3 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                dis_loss = cls_loss1 + cls_loss2 + cls_loss3
                weight =  self.softmax(self.mweight2(pred_src1)) 
                r21=weight[:, 0].reshape(pred_src.shape[0],1) 
                r22=weight[:, 1].reshape(pred_src.shape[0],1) 
                r23=weight[:, 2].reshape(pred_src.shape[0],1) 
                pred_src = r21*pred_src21 + r22*pred_src22 + r23*pred_src23 
                
                l1_loss = torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1) - torch.nn.functional.softmax(pred_tgt1, dim=1))
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_tgt2, dim=1) - torch.nn.functional.softmax(pred_tgt3, dim=1))                
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_src, dim=1) - torch.nn.functional.softmax(pred_src1, dim=1))                               
                l1_loss = torch.mean(l1_loss)
                
                cls_loss = F.nll_loss(F.log_softmax(pred_src21, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src22, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src23, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                if epo > pse_num:
                    data_st = self.sharedNet(data_srct)
                    data_tgt_p = self.avgpool(data_st)
                    data_tgt_p = data_tgt_p.view(data_tgt_p.size(0), -1)

                    data_stc1 = self.sonnetc1(data_st)
                    data_sts1 = self.sonnets1(data_st)
                    data_st1 = self.sonnet1(data_st)
                    data_stc1 = self.avgpool(data_stc1)
                    data_stc1 = data_stc1.view(data_stc1.size(0), -1)
                    data_sts1 = self.avgpool(data_sts1)
                    data_sts1 = data_sts1.view(data_sts1.size(0), -1)
                    
                    data_stc2 = self.sonnetc2(data_st)
                    data_sts2 = self.sonnets2(data_st)
                    data_st2 = self.sonnet2(data_st)
                    data_stc2 = self.avgpool(data_stc2)
                    data_stc2 = data_stc2.view(data_stc2.size(0), -1)
                    data_sts2 = self.avgpool(data_sts2)
                    data_sts2 = data_sts2.view(data_sts2.size(0), -1)
                    
                    data_stc3 = self.sonnetc3(data_st)
                    data_sts3 = self.sonnets3(data_st)
                    data_st3 = self.sonnet3(data_st)
                    data_stc3 = self.avgpool(data_stc3)
                    data_stc3 = data_stc3.view(data_stc3.size(0), -1)
                    data_sts3 = self.avgpool(data_sts3)
                    data_sts3 = data_sts3.view(data_sts3.size(0), -1)

                    pred_st11 = self.cls_fc_son11(data_stc1)
                    pred_st12 = self.cls_fc_son12(data_sts1)
                    pred_st13 = self.cls_fc_son13(data_st1)
                    pred_data = torch.stack([pred_st11, pred_st12, pred_st13], 2)
                    pred_st1 = self.member1(pred_data)
                    pred_st1 = torch.squeeze(pred_st1,2)

                    weight =  self.softmax(self.mweight1(pred_st1)) 
                    r11=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r12=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r13=weight[:, 2].reshape(data_st1.shape[0],1) 
                    pred_st1 = r11*pred_st11 + r12*pred_st12 + r13*pred_st13                  

                    pred_st21 = self.cls_fc_son21(data_stc2)
                    pred_st22 = self.cls_fc_son22(data_sts2)
                    pred_st23 = self.cls_fc_son23(data_st2)
                    pred_data = torch.stack([pred_st21, pred_st22, pred_st23], 2)
                    pred_st2 = self.member2(pred_data)
                    pred_st2 = torch.squeeze(pred_st2,2)

                    weight =  self.softmax(self.mweight2(pred_st2)) 
                    r21=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r22=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r23=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st2 = r21*pred_st21 + r22*pred_st22 + r23*pred_st23

                    cls_losst = F.nll_loss(F.log_softmax(pred_st21, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st22, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st23, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st2, dim=1), label_srct)
                    
                    pred_st31 = self.cls_fc_son31(data_stc3)
                    pred_st32 = self.cls_fc_son32(data_sts3)
                    pred_st33 = self.cls_fc_son33(data_st3)
                    pred_data = torch.stack([pred_st31, pred_st32, pred_st33], 2)
                    pred_st3 = self.member3(pred_data)
                    pred_st3 = torch.squeeze(pred_st3,2)

                    weight =  self.softmax(self.mweight3(pred_st3)) 
                    r31=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r32=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r33=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st3 = r31*pred_st31 + r32*pred_st32 + r33*pred_st33
                    
                    member = self.softmax(self.domain(data_tgt_p))
                    w1=member[:, 0].reshape(pred_st1.shape[0],1) 
                    w2=member[:, 1].reshape(pred_st1.shape[0],1)
                    w3=member[:, 2].reshape(pred_st1.shape[0],1)
                    cls_losst += F.nll_loss(F.log_softmax(w1*pred_st1+w2*pred_st2+w3*pred_st3, dim=1), label_srct)
                    
                    
                st_loss += mmd.mmd(data_srcc2, data_tgtc2)
                st_loss += mmd.mmd(data_srcs2, data_tgts2)
                st_loss += mmd.mmd(data_src2, data_tgt2)                                                          
                
                return st_loss, cls_loss + cls_losst, l1_loss/3, domain_loss/3+dis_loss/3
                
            if mark == 3:

                data_srcc3 = self.sonnetc3(data_src3)
                data_srcs3 = self.sonnets3(data_src3)
                data_src3 = self.sonnet3(data_src3)
                data_srcc3 = self.avgpool(data_srcc3)
                data_srcc3 = data_srcc3.view(data_srcc3.size(0), -1)
                data_srcs3 = self.avgpool(data_srcs3)
                data_srcs3 = data_srcs3.view(data_srcs3.size(0), -1)                                                                              
                
                pred_src31 = self.cls_fc_son31(data_srcc3)
                pred_src32 = self.cls_fc_son32(data_srcs3)
                pred_src33 = self.cls_fc_son33(data_src3)
                pred_src = torch.stack([pred_src31, pred_src32, pred_src33], 2)
                pred_src = self.member3(pred_src)
                pred_src1 = torch.squeeze(pred_src,2)

                pred_domain = self.mweight3(pred_src31)
                s_label = torch.zeros(pred_src.shape[0]).long().cuda()
                cls_loss1 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight3(pred_src32)
                s_label = torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss2 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                pred_domain = self.mweight3(pred_src33)
                s_label = 2*torch.ones(pred_src.shape[0]).long().cuda()
                cls_loss3 = F.nll_loss(F.log_softmax(pred_domain, dim=1), s_label)
                dis_loss = cls_loss1 + cls_loss2 + cls_loss3
                weight =  self.softmax(self.mweight3(pred_src1)) 
                r31=weight[:, 0].reshape(pred_src.shape[0],1) 
                r32=weight[:, 1].reshape(pred_src.shape[0],1) 
                r33=weight[:, 2].reshape(pred_src.shape[0],1) 
                pred_src = r31*pred_src31 + r32*pred_src32 + r33*pred_src33 
                
                l1_loss = torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1) - torch.nn.functional.softmax(pred_tgt1, dim=1))
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_tgt3, dim=1) - torch.nn.functional.softmax(pred_tgt2, dim=1))                
                l1_loss += torch.abs(torch.nn.functional.softmax(pred_src, dim=1) - torch.nn.functional.softmax(pred_src1, dim=1))                               
                l1_loss = torch.mean(l1_loss)
                
                cls_loss = F.nll_loss(F.log_softmax(pred_src31, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src32, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src33, dim=1), label_src)
                cls_loss += F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                if epo > pse_num:
                    data_st = self.sharedNet(data_srct)
                    data_tgt_p = self.avgpool(data_st)
                    data_tgt_p = data_tgt_p.view(data_tgt_p.size(0), -1)

                    data_stc1 = self.sonnetc1(data_st)
                    data_sts1 = self.sonnets1(data_st)
                    data_st1 = self.sonnet1(data_st)
                    data_stc1 = self.avgpool(data_stc1)
                    data_stc1 = data_stc1.view(data_stc1.size(0), -1)
                    data_sts1 = self.avgpool(data_sts1)
                    data_sts1 = data_sts1.view(data_sts1.size(0), -1)
                    
                    data_stc2 = self.sonnetc2(data_st)
                    data_sts2 = self.sonnets2(data_st)
                    data_st2 = self.sonnet2(data_st)
                    data_stc2 = self.avgpool(data_stc2)
                    data_stc2 = data_stc2.view(data_stc2.size(0), -1)
                    data_sts2 = self.avgpool(data_sts2)
                    data_sts2 = data_sts2.view(data_sts2.size(0), -1)
                    
                    data_stc3 = self.sonnetc3(data_st)
                    data_sts3 = self.sonnets3(data_st)
                    data_st3 = self.sonnet3(data_st)
                    data_stc3 = self.avgpool(data_stc3)
                    data_stc3 = data_stc3.view(data_stc3.size(0), -1)
                    data_sts3 = self.avgpool(data_sts3)
                    data_sts3 = data_sts3.view(data_sts3.size(0), -1)

                    pred_st11 = self.cls_fc_son11(data_stc1)
                    pred_st12 = self.cls_fc_son12(data_sts1)
                    pred_st13 = self.cls_fc_son13(data_st1)
                    pred_data = torch.stack([pred_st11, pred_st12, pred_st13], 2)
                    pred_st1 = self.member1(pred_data)
                    pred_st1 = torch.squeeze(pred_st1,2)
                    
                    weight =  self.softmax(self.mweight1(pred_st1)) 
                    r11=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r12=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r13=weight[:, 2].reshape(data_st1.shape[0],1) 
                    pred_st1 = r11*pred_st11 + r12*pred_st12 + r13*pred_st13                   

                    pred_st21 = self.cls_fc_son21(data_stc2)
                    pred_st22 = self.cls_fc_son22(data_sts2)
                    pred_st23 = self.cls_fc_son23(data_st2)
                    pred_data = torch.stack([pred_st21, pred_st22, pred_st23], 2)
                    pred_st2 = self.member2(pred_data)
                    pred_st2 = torch.squeeze(pred_st2,2)
                    
                    weight =  self.softmax(self.mweight2(pred_st2)) 
                    r21=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r22=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r23=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st2 = r21*pred_st21 + r22*pred_st22 + r23*pred_st23                   
                    
                    pred_st31 = self.cls_fc_son31(data_stc3)
                    pred_st32 = self.cls_fc_son32(data_sts3)
                    pred_st33 = self.cls_fc_son33(data_st3)
                    pred_data = torch.stack([pred_st31, pred_st32, pred_st33], 2)
                    pred_st3 = self.member3(pred_data)
                    pred_st3 = torch.squeeze(pred_st3,2)
                    
                    weight =  self.softmax(self.mweight3(pred_st3)) 
                    r31=weight[:, 0].reshape(pred_st1.shape[0],1) 
                    r32=weight[:, 1].reshape(pred_st1.shape[0],1) 
                    r33=weight[:, 2].reshape(pred_st1.shape[0],1) 
                    pred_st3 = r31*pred_st31 + r32*pred_st32 + r33*pred_st33

                    cls_losst = F.nll_loss(F.log_softmax(pred_st31, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st32, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st33, dim=1), label_srct)
                    cls_losst += F.nll_loss(F.log_softmax(pred_st3, dim=1), label_srct)
                    
                    member = self.softmax(self.domain(data_tgt_p))
                    w1=member[:, 0].reshape(pred_st1.shape[0],1) 
                    w2=member[:, 1].reshape(pred_st1.shape[0],1)
                    w3=member[:, 2].reshape(pred_st1.shape[0],1)
                    cls_losst += F.nll_loss(F.log_softmax(w1*pred_st1+w2*pred_st2+w3*pred_st3, dim=1), label_srct)
                    
                    
                st_loss += mmd.mmd(data_srcc3, data_tgtc3)
                st_loss += mmd.mmd(data_srcs3, data_tgts3)
                st_loss += mmd.mmd(data_src3, data_tgt3)  
                                          
                
                return st_loss, cls_loss + cls_losst, l1_loss/3, domain_loss/3+dis_loss/3

        else:
            
            data = self.sharedNet(data_src1)
            datap = self.avgpool(data)
            datap = datap.view(datap.size(0), -1)
            
            member = self.softmax(self.domain(datap))
            w1=member[:, 0].reshape(data.shape[0],1) 
            w2=member[:, 1].reshape(data.shape[0],1)
            w3=member[:, 2].reshape(data.shape[0],1)

            feac = self.sonnetc1(data)
            feas = self.sonnets1(data)
            fea1 = self.sonnet1(data)
            feac = self.avgpool(feac)
            feac = feac.view(feac.size(0), -1)
            feas = self.avgpool(feas)
            feas = feas.view(feas.size(0), -1)

            pred11 = self.cls_fc_son11(feac)
            pred12 = self.cls_fc_son12(feas)
            pred13 = self.cls_fc_son13(fea1)
            pred_src = torch.stack([pred11, pred12, pred13], 2)
            pred_src = self.member1(pred_src)
            pred1 = torch.squeeze(pred_src,2)

            weight =  self.softmax(self.mweight1(pred1)) 
            r1=weight[:, 0].reshape(pred1.shape[0],1) 
            r2=weight[:, 1].reshape(pred1.shape[0],1) 
            r3=weight[:, 2].reshape(pred1.shape[0],1) 
            pred1 = r1*pred11 + r2*pred12 + r3*pred13 

            feac = self.sonnetc2(data)
            feas = self.sonnets2(data)
            fea2 = self.sonnet2(data)
            feac = self.avgpool(feac)
            feac = feac.view(feac.size(0), -1)
            feas = self.avgpool(feas)
            feas = feas.view(feas.size(0), -1)
            
            
            pred21 = self.cls_fc_son21(feac)
            pred22 = self.cls_fc_son22(feas)
            pred23 = self.cls_fc_son23(fea2)
            pred_src = torch.stack([pred21, pred22, pred23], 2)
            pred_src = self.member2(pred_src)
            pred2 = torch.squeeze(pred_src,2)

            weight =  self.softmax(self.mweight2(pred2)) 
            r1=weight[:, 0].reshape(pred2.shape[0],1) 
            r2=weight[:, 1].reshape(pred2.shape[0],1) 
            r3=weight[:, 2].reshape(pred2.shape[0],1) 
            pred2 = r1*pred21 + r2*pred22 + r3*pred23  
            
            feac = self.sonnetc3(data)
            feas = self.sonnets3(data)
            fea3 = self.sonnet3(data)
            feac = self.avgpool(feac)
            feac = feac.view(feac.size(0), -1)
            feas = self.avgpool(feas)
            feas = feas.view(feas.size(0), -1)
                        
            pred31 = self.cls_fc_son31(feac)
            pred32 = self.cls_fc_son32(feas)
            pred33 = self.cls_fc_son33(fea3)
            pred_src = torch.stack([pred31, pred32, pred33], 2)
            pred_src = self.member3(pred_src)
            pred3 = torch.squeeze(pred_src,2) 

            weight =  self.softmax(self.mweight3(pred3)) 
            r1=weight[:, 0].reshape(pred3.shape[0],1) 
            r2=weight[:, 1].reshape(pred3.shape[0],1) 
            r3=weight[:, 2].reshape(pred3.shape[0],1) 
            pred3 = r1*pred31 + r2*pred32 + r3*pred33
            
            pred = w1*pred1+w2*pred2+w3*pred3                        
            
            return pred1, pred2, pred3, pred, [pred11, pred12, pred13], [pred21, pred22, pred23], [pred31, pred32, pred33]
    
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
