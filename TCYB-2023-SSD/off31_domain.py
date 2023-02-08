from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import pandas as pd
import math
import random
import data_loader
#import mydata_loader as data_loader
import SSD2 as models
#import MDAFuz as models
from torch.utils import model_zoo
import numpy as np
import mmd#_pdist as mmd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#'''
modelroot='./tramodels'
dataname = 'off31'
datapath = "./dataset/office31/"
domains = ['amazon','dslr','webcam'] 
#task = [0,1,2] 
task = [2,1,0]
#task = [0,2,1] 
num_classes=31
domain_num = 2

classpath = datapath + domains[0] + '/'
classlist = os.listdir(classpath)
classlist.sort()
#'''
'''
modelroot='./tramodels'
dataname = 'clef'
datapath = "./dataset/CLEF/"
domains = ['p','c','i'] 
#pc-i,ic-p, ip-c: 
#task = [0,1,2] 
task = [2,1,0]
#task = [2,0,1] 
num_classes = 12
domain_num = 2
'''

sam_flag = 0
list_samtrain = ['List','sel']
list_samtest = ['List','sel']
sel_samtra = list_samtrain[sam_flag]
sel_samtes = list_samtest[sam_flag]

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--iter', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=8, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default=datapath,
                    help='the path to load the data')
parser.add_argument('--source1_dir', type=str, default=domains[task[0]],#Art  Clipart   Product   Real World
                    help='the name of the source dir')
parser.add_argument('--source2_dir', type=str, default=domains[task[1]],
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default=domains[task[2]],
                    help='the name of the test dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
tra_flag = 1
rate = [0.2,0.8,0.6]
#'''
if tra_flag == 1:
    source1_loader = data_loader.load_image_TSS_val(args.root_path, args.source1_dir, classlist, sel_samtra, rate[task[0]], args.batch_size, kwargs)
    source2_loader = data_loader.load_image_TSS_val(args.root_path, args.source2_dir, classlist, sel_samtra, rate[task[1]], args.batch_size, kwargs)
    
    weight_loader = data_loader.load_image_test(args.root_path, args.test_dir, classlist, args.batch_size, kwargs)

if tra_flag == 0:
    weight_loader = data_loader.load_image_select(args.root_path, args.test_dir, classlist, args.batch_size, kwargs)
#'''
'''
if tra_flag == 1:
    source1_loader = data_loader.load_CLEF_TSS_val(args.root_path, args.source1_dir, sel_samtra, args.batch_size, kwargs)
    source2_loader = data_loader.load_CLEF_TSS_val(args.root_path, args.source2_dir, sel_samtra, args.batch_size, kwargs)
    
    weight_loader = data_loader.load_imageclef_test(args.root_path, args.test_dir, args.batch_size, kwargs)

if tra_flag == 0:
    weight_loader = data_loader.load_imageclef_select(args.root_path, args.test_dir, args.batch_size, kwargs)

'''
test_result = []

K = 1 # training times
train_tags = ['base_sf_alex', 'domain_alex']
train_flag = 1
train_tag = train_tags[train_flag]
load_tag = train_tags[0]

'''
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
'''
'''
np.random.seed(seed)
random.seed(seed)
'''


def train(traepo, model):
    source1_iter = iter(source1_loader['tra'])
    source2_iter = iter(source2_loader['tra'])
    
    s_num = [len(source1_loader['tra'].dataset), len(source2_loader['tra'].dataset)]
    s_num = np.array(s_num)
    
    effective_num = 1.0 - np.power(0.99, s_num)
    weights = (1.0 - 0.99) / np.array(effective_num)
    print(weights)
    weights = weights / np.sum(weights)
    print(weights)

    correct = 0
    early_stop = 1500
    

    for i in range(1, args.iter + 1):#i=100
        model.train()#model.train(False)
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.domain.parameters(), 'lr': LEARNING_RATE},#
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
	
        try:
            source_data1, source_label1 = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader['tra'])
            source_data1, source_label1 = source1_iter.next()
        try:
            source_data2, source_label2 = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader['tra'])
            source_data2, source_label2 = source2_iter.next()
        
        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()

        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)        

        optimizer.zero_grad()

        cls_loss = model(source_data1, source_data2) #cls_loss1, cls_loss2 
        cls_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                i, 100. * i / args.iter, cls_loss.item()))
    
        if i % (args.log_interval * 10) == 0:#
            w = weight(model)
            w = np.array(w)
            np.save('./MDA1/{}_{}_weight_alex.npy'.format(dataname, args.test_dir), w)
            w1 = w[:,0]
            w2 = w[:,1]
            count = np.zeros(domain_num)
            for j in range(len(w)):#i=0
                if w1[j]>w2[j]:
                    count[0] = count[0]+1
                if w2[j]>w1[j]:
                    count[1] = count[1]+1
            print(count)
            t = test(traepo, model)
            #t_correct1, t_accu1, t_correct2, t_accu2, t_correct, t_accu, t_correctw, t_accuw, t_correctm, t_accum= test(traepo, model, w1, w2)

            t_correct = t
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag))            
            print( "Cluster %s max correct:" % t, "\n")

            test_result.append(t)
            np.savetxt('./MDA1/{}_{}_{}.csv'.format(dataname, args.test_dir, train_tag), np.array(test_result), fmt='%.4f', delimiter=',')
            
        if i > early_stop:
            break
                
def test(traepo, model):#sort(w)
    model.eval()
    t_loss = 0
    correct1 = 0
    correct2 = 0

    
    for data, target in source1_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1) 
        label = torch.zeros(pred.shape[0]).long().cuda()                 
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), label).item()
        pred = pred.data.max(1)[1]
        correct1 += pred.eq(label.data.view_as(pred)).cpu().sum()       

    t_loss /= len(source1_loader['val'].dataset)  
    t_loss1=t_loss 
    t_loss = 0
    for data, target in source2_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1) 
        label = torch.ones(pred.shape[0]).long().cuda()                 
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), label).item()
        pred = pred.data.max(1)[1]
        correct2 += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    t_loss /= len(source2_loader['val'].dataset)  
    t_loss2=t_loss  
    print('\nloss1 {:.6f}, loss2 {:.6f}'.format(t_loss1, t_loss2))  
    print('\nsource1 {}/{}, source2 {}/{}'.format(correct1, len(source1_loader['val'].dataset), correct2, len(source2_loader['val'].dataset)))
          
    return correct1+correct2

def weight(model):#sort(w)
    model.eval()
    weight = []
    
    for data, target in weight_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1)
        w = np.array(pred.data.cpu())
        for j in range(pred.shape[0]):
            weight.append([w[j,0], w[j,1]])

    
    return weight
if __name__ == '__main__':

    traepo = 0
    if tra_flag == 1:
        model = models.MDAdomnet(domain_num)
        if args.cuda:
            model.cuda()
        save_model=torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, load_tag))		
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        train(traepo, model)
        print('The {} time trainging done!'.format(traepo+1))
    
    if tra_flag == 0:
        model = models.MDAdomnet_alex(domain_num)
        if args.cuda:
            model.cuda()
        model.load_state_dict(torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag)))  
        w = weight(model)
        w=np.array(w)
        np.save('./MDA1/{}_{}_weight_alex.npy'.format(dataname, args.test_dir), w)
        #w =  np.load('./MDA1/{}_{}_weight.npy'.format(dataname, args.test_dir))   
        w1 = w[:,0] 
        w2 = w[:,1]
        count = np.zeros(domain_num)
        for i in range(len(w)):#i=0
            if w1[i]>w2[i]:
                count[0] = count[0]+1
            if w2[i]>w1[i]:
                count[1] = count[1]+1                     
        print(count)
