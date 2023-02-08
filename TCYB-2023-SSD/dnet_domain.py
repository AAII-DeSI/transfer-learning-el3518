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
import SSD5 as models
#import MDAFuz as models
from torch.utils import model_zoo
import numpy as np
import mmd#_pdist as mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
modelroot='./tramodels'
dataname = 'offh'
datapath = "./dataset/OfficeHome/"
domains = ['Art','Clipart','Product', 'RealWorld'] 
#acp-r,acr-p, apr-c, cpr-a: 012-3,013-2,023-1,123-0
#task = [0,1,2,3] 
#task = [0,1,3,2]
#task = [0,2,3,1] 
task = [1,2,3,0]
num_classes = 65
'''
#'''
modelroot='./tramodels'
dataname = 'dnet'
datapath = "/scratch/DomainNet/"
domains = ['clipart','infograph','painting', 'quickdraw', 'real', 'sketch'] 
#ipqrs-c,cpqrs-i, ciqrs-p, ciprs-q, cipqs-r, cipqr-s
#task = [1,2,3,4,5,0] 
#task = [0,2,3,4,5,1]
#task = [0,1,3,4,5,2] 
#task = [0,1,2,4,5,3]
#task = [0,1,2,3,5,4]
task = [0,1,2,3,4,5]

num_classes = 345
#'''

domain_num = 5

classpath = datapath + domains[0] + '/'
classlist = os.listdir(classpath)
classlist.sort()

sam_flag = 1
list_samtrain = ['_train','sel']
list_samtest = ['_test','sel']
sel_samtra = list_samtrain[sam_flag]
sel_samtes = list_samtest[sam_flag-1]


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--iter', type=int, default=15000, metavar='N',
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
parser.add_argument('--source1_dir', type=str, default=domains[task[0]],
                    help='the name of the source dir')
parser.add_argument('--source2_dir', type=str, default=domains[task[1]],
                    help='the name of the source dir')
parser.add_argument('--source3_dir', type=str, default=domains[task[2]],
                    help='the name of the source dir')                    
parser.add_argument('--source4_dir', type=str, default=domains[task[3]],
                    help='the name of the source dir')
parser.add_argument('--source5_dir', type=str, default=domains[task[4]],
                    help='the name of the source dir') 
parser.add_argument('--test_dir', type=str, default=domains[task[5]],
                    help='the name of the test dir')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
tra_flag = 1
rate = [0.6,0.6,0.7,0.75,0.75,0.7,0.6]

if tra_flag == 1:
    source1_loader = data_loader.load_image_TSS_val(args.root_path, args.source1_dir, classlist, sel_samtra, rate[task[0]], args.batch_size, kwargs)
    source2_loader = data_loader.load_image_TSS_val(args.root_path, args.source2_dir, classlist, sel_samtra, rate[task[1]], args.batch_size, kwargs)
    source3_loader = data_loader.load_image_TSS_val(args.root_path, args.source3_dir, classlist, sel_samtra, rate[task[2]], args.batch_size, kwargs)
    source4_loader = data_loader.load_image_TSS_val(args.root_path, args.source4_dir, classlist, sel_samtra, rate[task[3]], args.batch_size, kwargs)    
    source5_loader = data_loader.load_image_TSS_val(args.root_path, args.source5_dir, classlist, sel_samtra, rate[task[4]], args.batch_size, kwargs)    
    
    
    source1_num = len(source1_loader['val'].dataset)
    source2_num = len(source2_loader['val'].dataset)
    source3_num = len(source3_loader['val'].dataset)
    source4_num = len(source4_loader['val'].dataset)
    source5_num = len(source5_loader['val'].dataset)
    weight_loader = data_loader.load_image_TSS_select(args.root_path, args.test_dir, classlist, '_test', args.batch_size, kwargs)

if tra_flag == 0:
    weight_loader = data_loader.load_image_TSS_test(args.root_path, args.test_dir, classlist, sel_samtst, args.batch_size, kwargs)

test_result = []
train_loss = []
test_loss = []

K = 1 # training times
train_tags = ['base_sf1', 'domain']
train_flag = 1
train_tag = train_tags[train_flag]
load_tag = train_tags[0]
#vote = []

def train(traepo,model):
    source1_iter = iter(source1_loader['tra'])
    source2_iter = iter(source2_loader['tra'])
    source3_iter = iter(source3_loader['tra'])
    source4_iter = iter(source4_loader['tra'])
    source5_iter = iter(source5_loader['tra'])
    
    s_num = [len(source1_loader['tra'].dataset), len(source2_loader['tra'].dataset), len(source3_loader['tra'].dataset), len(source4_loader['tra'].dataset), len(source5_loader['tra'].dataset)]
    s_num = np.array(s_num)
    
    effective_num = 1.0 - np.power(0.999, s_num)
    weights = (1.0 - 0.999) / np.array(effective_num)
    print(weights)
    weights = weights / np.sum(weights)
    print(weights)

    correct = 0
    early_stop = 5000

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
        try:
            source_data3, source_label3 = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader['tra'])
            source_data3, source_label3 = source3_iter.next()
        try:
            source_data4, source_label4 = source4_iter.next()
        except Exception as err:
            source4_iter = iter(source4_loader['tra'])
            source_data4, source_label4 = source4_iter.next()
        try:
            source_data5, source_label5 = source5_iter.next()
        except Exception as err:
            source5_iter = iter(source5_loader['tra'])
            source_data5, source_label5 = source5_iter.next()
        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            source_data3, source_label3 = source_data3.cuda(), source_label3.cuda()
            source_data4, source_label4 = source_data4.cuda(), source_label4.cuda()
            source_data5, source_label5 = source_data5.cuda(), source_label5.cuda()

        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2) 
        source_data3, source_label3 = Variable(source_data3), Variable(source_label3)  
        source_data4, source_label4 = Variable(source_data4), Variable(source_label4) 
        source_data5, source_label5 = Variable(source_data5), Variable(source_label5)      

        optimizer.zero_grad()

        cls_loss = model(source_data1, source_data2, source_data3, source_data4, source_data5)
        cls_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                i, 100. * i / args.iter, cls_loss.item()))
                         
        if i % (args.log_interval * 10) == 0:#
            t = test(traepo, model)
            t_correct = t
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag))     

            w = weight(model)
            w = np.array(w)
            np.save('./MDA1/{}_{}_weight.npy'.format(dataname, args.test_dir), w)
        
            w1 = w[:,0] 
            w2 = w[:,1]
            w3 = w[:,2]
            w4 = w[:,3]
            w5 = w[:,4]
            count = np.zeros(domain_num)
            for j in range(len(w)):#i=0
                idx_sort = np.argsort(-w[j,:])
                count[idx_sort[0]] = count[idx_sort[0]]+1
            print(count)
            
        if i > early_stop:
            break
                
def test(traepo, model):#sort(w)
    model.eval()
    t_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0

    
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
      
    t_loss = 0
    for data, target in source3_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) 
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1) 
        label = 2*torch.ones(pred.shape[0]).long().cuda()                 
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), label).item()
        pred = pred.data.max(1)[1]
        correct3 += pred.eq(label.data.view_as(pred)).cpu().sum()
    t_loss /= len(source3_loader['val'].dataset)  
    t_loss3=t_loss 
    
    t_loss = 0
    for data, target in source4_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) 
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1) 
        label = 3*torch.ones(pred.shape[0]).long().cuda()                 
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), label).item()
        pred = pred.data.max(1)[1]
        correct4 += pred.eq(label.data.view_as(pred)).cpu().sum()
    t_loss /= len(source4_loader['val'].dataset)  
    t_loss4=t_loss 
    
    t_loss = 0
    for data, target in source5_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target) 
        pred = model(data)       
        pred = torch.nn.functional.softmax(pred, dim=1) 
        label = 4*torch.ones(pred.shape[0]).long().cuda()                 
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), label).item()
        pred = pred.data.max(1)[1]
        correct5 += pred.eq(label.data.view_as(pred)).cpu().sum()
    t_loss /= len(source5_loader['val'].dataset)  
    t_loss5=t_loss 
    
    print('\nloss1 {:.6f}, loss2 {:.6f}, loss3 {:.6f}, loss4 {:.6f}, loss5 {:.6f}'.format(t_loss1, t_loss2, t_loss3, t_loss4, t_loss5))  
    print('\nsource1 {}/{}, source2 {}/{}, source3 {}/{}, source4 {}/{}, source5 {}/{}'.format(correct1, source1_num, correct2, source2_num, correct3, source3_num, correct4, source4_num, correct5, source5_num))
          
    return correct1+correct2+correct3+correct4+correct5

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
            weight.append([w[j,0], w[j,1], w[j,2], w[j,3], w[j,4]])

    
    return weight

if __name__ == '__main__':
    
    traepo = 0
    if tra_flag == 1:
        model = models.MDAdomnet(domain_num)
        if args.cuda:
            model.cuda()
        #save_model=torch.load('{}/{}_{}_MDA_{}_{}{}.pth'.format(modelroot, dataname, args.test_dir, load_tag, load_sel, traepo))
        save_model=torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, load_tag))		
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        train(traepo, model)
        print('The {} time trainging done!'.format(traepo+1))
    
    
    if tra_flag == 0:
        model = models.MDAdomnet(domain_num)
        if args.cuda:
            model.cuda()
        #save_model=torch.load('{}/pretrain/{}_{}_MDA_{}_{}{}.pth'.format(modelroot, dataname, args.test_dir, load_tag, load_sel, traepo))

        model.load_state_dict(torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag)))
        w = weight(model)
        weight = np.array(w)
        np.save('./MDA1/{}_{}_weight.npy'.format(dataname, args.test_dir), w)
        
        w1 = weight.ix[:,0] 
        w2 = weight.ix[:,1]
        w3 = weight.ix[:,2]
        count = np.zeros(domain_num)
        for i in range(len(w)):#i=0
            if w1[i]>w2[i] and w1[i]>w3[i]:
                count[0] = count[0]+1
            if w2[i]>w1[i] and w2[i]>w3[i]:
                count[1] = count[1]+1
            if w3[i]>w2[i] and w3[i]>w1[i]:
                count[2] = count[2]+1
        print(count)
    
    
     
        
        
