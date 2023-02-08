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
import SSD3 as models
#import MDAFuz as models
from torch.utils import model_zoo
import numpy as np
import mmd#_pdist as mmd
from utilities import BCELossForMultiClassification, AccuracyCounter, variable_to_numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

modelroot='./tramodels'
dataname = 'offh'
datapath = "./dataset/OfficeHome/"
domains = ['Art','Clipart','Product', 'RealWorld'] 
#acp-r,acr-p, apr-c, cpr-a: 012-3,013-2,023-1,123-0
task = [0,1,2,3] 
#task = [0,1,3,2]
#task = [0,2,3,1] 
#task = [1,2,3,0]

path = 'MDA1'

num_classes = 65

classpath = datapath + domains[0] + '/'
classlist = os.listdir(classpath)
classlist.sort()

list_samtrain = ['List','tra']
list_samtest = ['List','tst']
sel_samtra = list_samtrain[1]


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
parser.add_argument('--num_class', type=int, default=num_classes, metavar='N',
                    help='number of classes')
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
parser.add_argument('--test_dir', type=str, default=domains[task[3]],
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
sel_flag = 1
source_flag = 1
source_dir = domains[task[source_flag-1]]

if sel_flag == 0:
    target_loader = data_loader.load_image_TSS_val(args.root_path, args.test_dir, classlist, sel_samtra, args.batch_size, kwargs)
    source1_loader = data_loader.load_image_test(args.root_path, args.source1_dir, classlist, args.batch_size, kwargs)    
    source2_loader = data_loader.load_image_test(args.root_path, args.source2_dir, classlist, args.batch_size, kwargs)    
    source3_loader = data_loader.load_image_test(args.root_path, args.source3_dir, classlist, args.batch_size, kwargs)    
    
    test_num = len(target_loader['val'].dataset)

if sel_flag == 1:
    source_select_loader = data_loader.load_image_select(args.root_path, source_dir, classlist, args.batch_size, kwargs)
    source_num = len(source_select_loader.dataset)


K = 1 # training times
train_tags = ['base_sf', 'select']
train_tag = train_tags[1]
load_tag = train_tags[0]

def train(traepo, model):
    target_iter = iter(target_loader['tra'])
    
    correct = 0
    early_stop = 2000

    for i in range(1, args.iter + 1):#i=100
        model.train()#model.train(False)
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.fcs.parameters(), 'lr': LEARNING_RATE},#mbfcs.
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)
                        

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader['tra'])
            target_data, target_label = target_iter.next()
        
        if args.cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()

        target_data, target_label = Variable(target_data), Variable(target_label)
        bin_label = F.one_hot(target_label, args.num_class)

        optimizer.zero_grad()

        pre_label = model(target_data)
        cls_loss = BCELossForMultiClassification(bin_label, pre_label)
        cls_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source iter: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                i, 100. * i / args.iter, cls_loss.item()))
                          
        if i % (args.log_interval * 10) == 0:#
            t = test(model)
            #t_correct1, t_accu1, t_correct2, t_accu2, t_correct, t_accu, t_correctw, t_accuw, t_correctm, t_accum= test(traepo, model, w1, w2)

            t_correct = t
            if t_correct >= correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag))            
            print( "Classification %s max acc:" % args.test_dir, correct/test_num, "\n")
            test_source(model)

        if i > early_stop:
            break
                
def test(model):#sort(w)
    model.eval()
    #t_loss = 0
    correct = 0    
    
    for data, target in target_loader['val']:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        bin_label = F.one_hot(target, args.num_class) 
        pred = model(data)
        #for i in range(args.num_class):
            #pred[i] = torch.nn.functional.softmax(pred[i], dim=1)
        correct += np.equal(np.argmax(variable_to_numpy(pred), 1), np.argmax(variable_to_numpy(bin_label), 1)).sum()
        
        '''
        counter = AccuracyCounter()
        counter.addOntBatch(variable_to_numpy(pred), variable_to_numpy(bin_label))
        correct = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()            
        '''    
        #correct += pred.eq(bin_label.data.view_as(pred)).cpu().sum()
              
    #t_loss /= len(source_test_loader.dataset)      
    #print('\nloss {:.6f}'.format(t_loss))  
    #print('\nacc {}'.format(correct)#\nsource {}/{}, source_num, correct
    print('\nsource {}/{}'.format(correct, test_num))
          
    return correct
    
def test_source(model):#sort(w)
    model.eval()
    #t_loss = 0
    correct = 0    
    
    for data, target in source1_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        bin_label = F.one_hot(target, args.num_class) 
        pred = model(data)
        #for i in range(args.num_class):
            #pred[i] = torch.nn.functional.softmax(pred[i], dim=1)
        correct += np.equal(np.argmax(variable_to_numpy(pred), 1), np.argmax(variable_to_numpy(bin_label), 1)).sum()
        
        '''
        counter = AccuracyCounter()
        counter.addOntBatch(variable_to_numpy(pred), variable_to_numpy(bin_label))
        correct = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()            
        '''    
        #correct += pred.eq(bin_label.data.view_as(pred)).cpu().sum()
    print('\nsource {}/{}'.format(correct, len(source1_loader.dataset)))
               
    correct = 0    
    
    for data, target in source2_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        bin_label = F.one_hot(target, args.num_class) 
        pred = model(data)
        #for i in range(args.num_class):
            #pred[i] = torch.nn.functional.softmax(pred[i], dim=1)
        correct += np.equal(np.argmax(variable_to_numpy(pred), 1), np.argmax(variable_to_numpy(bin_label), 1)).sum()
        
    
    #t_loss /= len(source_test_loader.dataset)      
    #print('\nloss {:.6f}'.format(t_loss))  
    #print('\nacc {}'.format(correct)#\nsource {}/{}, source_num, correct
    print('\nsource {}/{}'.format(correct, len(source2_loader.dataset)))
    
    correct = 0    
    
    for data, target in source3_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        bin_label = F.one_hot(target, args.num_class) 
        pred = model(data)
        #for i in range(args.num_class):
            #pred[i] = torch.nn.functional.softmax(pred[i], dim=1)
        correct += np.equal(np.argmax(variable_to_numpy(pred), 1), np.argmax(variable_to_numpy(bin_label), 1)).sum()
        
    
    #t_loss /= len(source_test_loader.dataset)      
    #print('\nloss {:.6f}'.format(t_loss))  
    #print('\nacc {}'.format(correct)#\nsource {}/{}, source_num, correct
    print('\nsource {}/{}'.format(correct, len(source3_loader.dataset)))


def select(model):#sort(w)
    model.eval()
    #t_loss = 0
    correct = 0    
    pre_pro = []
    pre_lab = []
    tar_lab = []
    for data, target in source_select_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        bin_label = F.one_hot(target, args.num_class)
        pred = model(data)
        correct += np.equal(np.argmax(variable_to_numpy(pred), 1), np.argmax(variable_to_numpy(bin_label), 1)).sum()
        
        '''
        counter = AccuracyCounter()
        counter.addOntBatch(variable_to_numpy(pred), variable_to_numpy(bin_label))
        correct = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda() 
        '''
        print('\nacc {}'.format(correct/source_num))#\nsource {}/{}, source_num, correct
     
        pre_label = np.argmax(np.array(pred.data.cpu()),1)
        #pre_prob = np.array(pred.data.max(1)[0].cpu())
        pre_lab.extend(pre_label)
        #pre_pro.extend(pre_prob)
        tar_lab.extend(np.array(target.data.cpu()))
        pre = np.array(pred.data.cpu())
        for j in range(pred.shape[0]):
            pre_pro.append(pre[j, target[j]])            
        
    return pre_pro, pre_lab, tar_lab

def sel_sam(prob, lab, num_classes):
    #prob = out_pred[:,0]
    #lab = out_pred[:,2]
    idex = []
    for cls in range(num_classes):
        idx = [idx for idx, lab in enumerate(lab) if lab == cls]           
        if len(idx) > 0:
            idxs = [idxs for idxs in idx if prob[idxs] >= np.median(prob[idx])]#-(np.median(prob[idx])-np.min(prob[idx]))/(args.sk+1)]
            idex.extend(idxs)    
    return idex

if __name__ == '__main__':

    #for traepo in range(K):
    if sel_flag == 0:
        traepo = 0       
        model = models.Discriminator(num_classes)
        if args.cuda:
            model.cuda()
        save_model=torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, load_tag))		
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        train(traepo, model)
        print('The {} time trainging done!'.format(traepo+1))
    
    if sel_flag == 1:
        traepo = 0
        model = models.Discriminator(num_classes)
        if args.cuda:
            model.cuda()
        model.load_state_dict(torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag)))    
        out = select(model)
        out_pred = []
        for i in range(len(out[0])):
            out_pred.append([out[0][i], out[1][i], out[2][i]])

        out_pred = np.array(out_pred)
        idex = sel_sam(out_pred[:,0], out_pred[:,2], num_classes)
        
        org_file = datapath + source_dir + 'List.txt'
        file_org = open(org_file, 'r').readlines()
        idxall = [idxall for idxall, item in enumerate(file_org)]
        
        out_file = datapath + source_dir + 'sel.txt'
        file = open(out_file,'w')                   
        for i in idex:
            lines = file_org[i]
            line = lines.strip().split(' ')
            new_lines = line[0]
            file.write('%s %s\n' % (new_lines, int(out_pred[i,2])))
        file.close()
        
        np.savetxt('./{}/{}_src_sel_{}_{}.csv'.format(path, dataname, args.test_dir, source_dir), np.array(out_pred), fmt='%.6f', delimiter=',')      
            

    

        
