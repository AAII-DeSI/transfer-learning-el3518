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
num_classes = 31

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
#task = [2,1,0]
task = [2,0,1] #
num_classes = 12
'''
list_sam = ['List', 'sel', 'tra', 'tst']
sel_samtra = list_sam[1]
sel_samtst = list_sam[0]
sel_samtar = list_sam[2]
sel_samttst = list_sam[3]

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
#'''
source1_loader = data_loader.load_image_TSS(args.root_path, args.source1_dir, classlist, sel_samtra, args.batch_size, kwargs)
source2_loader = data_loader.load_image_TSS(args.root_path, args.source2_dir, classlist, sel_samtra, args.batch_size, kwargs)

sourcet_loader = data_loader.load_image_TSS(args.root_path, args.test_dir, classlist, sel_samtar, args.batch_size, kwargs)
sourcetst_loader = data_loader.load_image_TSS(args.root_path, args.test_dir, classlist, sel_samttst, args.batch_size, kwargs)
    
target_train_loader = data_loader.load_image_train(args.root_path, args.test_dir, classlist, args.batch_size, kwargs)
target_test_loader = data_loader.load_image_test_idx(args.root_path, args.test_dir, classlist, args.batch_size, kwargs)
target_num = len(target_test_loader.dataset)
#'''
test_result = []
'''
source1_loader = data_loader.load_CLEF_TSS(args.root_path, args.source1_dir, sel_samtra, args.batch_size, kwargs)
source2_loader = data_loader.load_CLEF_TSS(args.root_path, args.source2_dir, sel_samtra, args.batch_size, kwargs)

sourcet_loader = data_loader.load_CLEF_TSS(args.root_path, args.test_dir, sel_samtar, args.batch_size, kwargs)
sourcetst_loader = data_loader.load_CLEF_TSS(args.root_path, args.test_dir, sel_samttst, args.batch_size, kwargs)
    
target_train_loader = data_loader.load_imageclef_train(args.root_path, args.test_dir, args.batch_size, kwargs)
target_test_loader = data_loader.load_imageclef_test_idx(args.root_path, args.test_dir, args.batch_size, kwargs)
target_num = len(target_test_loader.dataset)
'''
K = 1 # training times
train_tags = ['base_sf','tss_sf']
load_tag = train_tags[0]
train_tag = train_tags[1]


def train(traepo, model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    #'''
    sourcet_iter = iter(sourcet_loader)
    sourcetst_iter = iter(sourcetst_loader)
    #'''
    target_iter = iter(target_train_loader)
    correct = 0

    early_stop = 5000

    for i in range(1, args.iter + 1):#i=100
        model.train()#model.train(False)
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            #{'params': model.domain.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=args.momentum, weight_decay=args.l2_decay)

        try:
            source_data1, source_label1 = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data1, source_label1 = source1_iter.next()
        try:
            source_data2, source_label2 = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data2, source_label2 = source2_iter.next()
        #'''
        try:
            source_datat, source_labelt = sourcet_iter.next()
        except Exception as err:
            sourcet_iter = iter(sourcet_loader)
            source_datat, source_labelt = sourcet_iter.next()
        
        try:
            source_datatst, __ = sourcetst_iter.next()
        except Exception as err:
            sourcetst_iter = iter(sourcetst_loader)
            source_datatst, __ = sourcetst_iter.next()

        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            source_datat, source_labelt = source_datat.cuda(), source_labelt.cuda()
            source_datatst = source_datatst.cuda()
            target_data = target_data.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        source_datat, source_labelt = Variable(source_datat), Variable(source_labelt)
        source_datatst = Variable(source_datatst)      
        target_data = Variable(target_data)
        optimizer.zero_grad()

        
        #if i-1 == 0 or i % args.log_interval*10 == 0:
        #    tar_cen1, tar_cen2 = obtain_cen(sourcet_loader, model)
        
        domain_loss, class_loss, cls_loss, l1_loss = model(source_data1, 
                                                           source_datat, source_datatst, 
                                                           target_data, source_label1, 
                                                           source_labelt, mark=1, test=False)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1  #target_data, 
        loss1 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        #loss1 = cls_loss1 + cls_loss2 + cls_loss + gamma * l1_loss
        loss1.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tclass_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss1.item(), cls_loss.item(), domain_loss.item(), class_loss.item(), l1_loss.item()))
        
        domain_loss, class_loss, cls_loss, l1_loss  = model(source_data2, 
                                                            source_datat, source_datatst,
                                                            target_data, source_label2, 
                                                            source_labelt, mark=2, test=False)        
        
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss2 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        #loss2 = cls_loss1 + cls_loss2 + cls_loss + gamma * l1_loss
        loss2.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tclass_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss2.item(), cls_loss.item(), domain_loss.item(), class_loss.item(), l1_loss.item()))
   
        if i % (args.log_interval * 10) == 0:
            t_num, t_accu = test(traepo, model)

            t_correct = t_num[2]
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_{}{}.pth'.format(modelroot, dataname, args.test_dir, train_tag, traepo))            
            print( "Target %s max correct:" % args.test_dir, correct, "\n")

            t_num.extend(t_accu)
            test_result.append(t_num)
            np.savetxt('./MDA1/{}_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(test_result), fmt='%.4f', delimiter=',')
            
        if i > early_stop:
            break
                
def test(traepo, model):#sort(w)
    model.eval()
    t_lossw = 0
    correct1 = 0
    correct2 = 0
    correctw = 0
    correctm = 0
    w = torch.from_numpy(np.load('./MDA1/{}_{}_weight_alex.npy'.format(dataname, args.test_dir)))
    
    for data, target, idx in target_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred1, pred2 = model(data, test=True) #, pred
        
        weight = w[idx,:].cuda()
        w1=weight[:, 0].reshape(data.shape[0],1) 
        w2=weight[:, 1].reshape(data.shape[0],1)
        pred = w1*pred1 + w2*pred2
        t_lossw += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
        pred = pred.data.max(1)[1]
        correctw += pred.eq(target.data.view_as(pred)).cpu().sum()

        pred1 = torch.nn.functional.softmax(pred1, dim=1)
        pred2 = torch.nn.functional.softmax(pred2, dim=1)

        pred = pred1.data.max(1)[1]
        correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
      
        pred = (pred1 + pred2)/2
        pred = pred.data.max(1)[1]
        correctm += pred.eq(target.data.view_as(pred)).cpu().sum()

    t_lossw /= len(target_test_loader.dataset)

    accu1 = float(correct1) / len(target_test_loader.dataset)*100 
    accu2 = float(correct2) / len(target_test_loader.dataset)*100 
    accuw = float(correctw) / len(target_test_loader.dataset)*100 
    accum = float(correctm) / len(target_test_loader.dataset)*100 
    correct_num = [correct1, correct2, correctw, correctm]
    accu = [accu1, accu2, accuw, accum]
    print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            t_lossw, correctw, len(target_test_loader.dataset),
            100. * correctw / len(target_test_loader.dataset)))
    print('\nsource1 {}, source2 {}, mean {}'.format(correct1, correct2, correctm))
          
    return correct_num, accu

if __name__ == '__main__':
    #'''
    #for traepo in range(K):
    traepo = 0
    model = models.TSSnet(num_classes)      
    if args.cuda:
        model.cuda()       

    save_model=torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, load_tag))        
    model_dict = model.state_dict()        
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}        
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
   
    train(traepo, model)
    print('The {} time trainging done!'.format(traepo+1))
        

