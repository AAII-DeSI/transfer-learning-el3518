#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = Pdist2(total, total)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def cmmd(source, target, s_label, t_label, classes, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    s_label = s_label.cpu()
    s_label = s_label.view(32,1)
    s_label = torch.zeros(32, classes).scatter_(1, s_label.data, 1)
    s_label = Variable(s_label).cuda()

    t_label = t_label.cpu()
    t_label = t_label.view(32, 1)
    t_label = torch.zeros(32, classes).scatter_(1, t_label.data, 1)
    t_label = Variable(t_label).cuda()

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    loss += torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX +
                      torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY -
                      2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)
    return loss
    
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def euclid_dist(x, y):
    x_sq = (x ** 2).mean(-1)
    x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
    y_sq = (y ** 2).mean(-1)
    y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
    xy = torch.mm(x, y.t()) / x.size(-1)
    dist = x_sq_ + y_sq_ - 2 * xy

    return dist

def construct_adj(feats, mean, adj):
    dist = euclid_dist(mean, feats)
    sim = torch.exp(-dist / (2 * 0.005 ** 2))
    E = torch.eye(feats.shape[0]).float().cuda()

    A = torch.cat([adj, sim], dim = 1)
    B = torch.cat([sim.t(), E], dim = 1)
    gcn_adj = torch.cat([A, B], dim = 0)

    return gcn_adj
