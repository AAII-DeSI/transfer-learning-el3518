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
task = [2,1,0]
#task = [2,0,1] 
num_classes = 12
'''
list_sam = ['List']
sel_sam = list_sam[0]

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
sel_flag = 1
#'''
if sel_flag==0:
    source1_loader = data_loader.load_training(args.root_path, args.source1_dir, args.batch_size, kwargs)
    source2_loader = data_loader.load_training(args.root_path, args.source2_dir, args.batch_size, kwargs)

    target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_num = len(target_test_loader.dataset)

    test_result = []
if sel_flag==1:
    #source_dir = source1_dir
    source1_select_loader = data_loader.load_image_select(args.root_path, args.source1_dir, classlist, args.batch_size, kwargs)
    source2_select_loader = data_loader.load_image_select(args.root_path, args.source2_dir, classlist, args.batch_size, kwargs)
    target_select_loader = data_loader.load_image_select(args.root_path, args.test_dir, classlist, args.batch_size, kwargs)

#'''
'''
if sel_flag==0:
    source1_loader = data_loader.load_CLEF_TSS(args.root_path, args.source1_dir, sel_sam, args.batch_size, kwargs)
    source2_loader = data_loader.load_CLEF_TSS(args.root_path, args.source2_dir, sel_sam, args.batch_size, kwargs)

    target_test_loader = data_loader.load_imageclef_test(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_num = len(target_test_loader.dataset)
    test_result = []

if sel_flag==1:
    #source_dir = source1_dir
    source1_select_loader = data_loader.load_imageclef_select(args.root_path, args.source1_dir, args.batch_size, kwargs)
    source2_select_loader = data_loader.load_imageclef_select(args.root_path, args.source2_dir, args.batch_size, kwargs)
    target_select_loader = data_loader.load_imageclef_select(args.root_path, args.test_dir, args.batch_size, kwargs)
'''

   
K = 1 # training times
train_tags = ['base_sf_alex']
train_flag = 0
train_tag = train_tags[train_flag]
load_tag = train_tags[0]

def train(traepo,model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)

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

        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()

        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)        
        optimizer.zero_grad()

        cls_loss = model(source_data1, source_label1, mark=1)

        #gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        #loss1 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        cls_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tcls_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, cls_loss.item()))
        
        cls_loss = model(source_data2, source_label2, mark=2)
        
        #gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        #loss2 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        cls_loss.backward()
        optimizer.step()

        if i % args.log_interval == 0: #, l1_loss.item()   \tl1_Loss: {:.6f}
            print('Train source2 iter: {} [({:.0f}%)]\tcls_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, cls_loss.item()))

        if i % (args.log_interval * 10) == 0:
            t_num, t_accu = test(traepo, model)

            t_correct = t_num[2]
            t_acc = t_accu[2]
            
            if t_correct > correct:
                correct = t_correct
                acc = t_acc 
                torch.save(model.state_dict(), '{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag))            
            print( "Target: {}, Correct: {}, Accuracy: {:.4f}" .format(args.test_dir, correct, acc))

            t_num.extend(t_accu)
            test_result.append(t_num)
            np.savetxt('./MDA1/{}_test_{}_{}.csv'.format(dataname, args.test_dir, train_tag), np.array(test_result), fmt='%.4f', delimiter=',')
            
        if i > early_stop:
            break
                
def test(traepo, model):#sort(w)
    model.eval()
    t_loss = 0
    correct1 = 0
    correct2 = 0
    correct = 0
    
    for data, target in target_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred1, pred2 = model(data)

        pred1 = torch.nn.functional.softmax(pred1, dim=1)
        pred2 = torch.nn.functional.softmax(pred2, dim=1)

        pred = pred1.data.max(1)[1]
        correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = (pred1 + pred2)/2
        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
        pred = pred.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    t_loss /= len(target_test_loader.dataset)
   
    accu1 = float(correct1) / len(target_test_loader.dataset)*100 
    accu2 = float(correct2) / len(target_test_loader.dataset)*100 
    accu = float(correct) / len(target_test_loader.dataset)*100 

    correct_num = [correct1, correct2, correct]
    accu = [accu1, accu2, accu]
    print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            t_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
    print('\nsource1 {}, source2 {}'.format(correct1, correct2))
          
    return correct_num, accu

def select(traepo, model, domain_flag):#sort(w)
    model.eval()   
    #domain_flag = 0 #0-source; 1-target

    if domain_flag == 0: # labeled source 
        spre_label1 = []
        tag_label1 = []
        spre_label2 = []
        tag_label2 = []
        correct = 0
        for data, target in source1_select_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            pred,_ = model(data)        
            pred = torch.nn.functional.softmax(pred, dim=1)
            
            pro = np.array(pred.data.cpu())
            pro_pre = np.array(pred.data.max(1)[0].cpu())
            lab_pre = np.array(pred.data.max(1)[1].cpu())
            lab = np.array(target.data.cpu())
            
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()       

            print('Training accuracy: {:.4f} %'.format(float(correct) / len(source1_select_loader.dataset)*100))
                           
            #if domain_flag == 0: # labeled source
            for j in range(pred.shape[0]):
                #spre_label.append([pro[j]])#,pre[j],lab[j]])
                spre_label1.append([pro[j,lab[j]]])
                tag_label1.append([lab[j]])
            
        correct = 0
        for data, target in source2_select_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            _, pred = model(data)        
            pred = torch.nn.functional.softmax(pred, dim=1)
            
            pro = np.array(pred.data.cpu())
            pro_pre = np.array(pred.data.max(1)[0].cpu())
            lab_pre = np.array(pred.data.max(1)[1].cpu())
            lab = np.array(target.data.cpu())
            
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()       

            print('Training accuracy: {:.4f} %'.format(float(correct) / len(source2_select_loader.dataset)*100))
                           
            #if domain_flag == 0: # labeled source
            for j in range(pred.shape[0]):
                #spre_label.append([pro[j]])#,pre[j],lab[j]])
                spre_label2.append([pro[j,lab[j]]])
                tag_label2.append([lab[j]])

        
        return spre_label1, tag_label1, spre_label2, tag_label2
    if domain_flag == 1:
        spre_label = []
        tag_label = []
        correct1 = 0
        correct2 = 0
        for data, target in target_select_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            pred1, pred2 = model(data)        

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            #pred = (pred1 + pred2)/2
            
            pro1 = np.array(pred1.data.cpu())
            pro1_pre = np.array(pred1.data.max(1)[0].cpu())
            lab1_pre = np.array(pred1.data.max(1)[1].cpu())
            lab1 = np.array(target.data.cpu())
            
            pro2 = np.array(pred2.data.cpu())
            pro2_pre = np.array(pred2.data.max(1)[0].cpu())
            lab2_pre = np.array(pred2.data.max(1)[1].cpu())
            lab2 = np.array(target.data.cpu())
            
            pred1 = pred1.data.max(1)[1]
            correct1 += pred1.eq(target.data.view_as(pred1)).cpu().sum() 
            
            pred2 = pred2.data.max(1)[1]
            correct2 += pred2.eq(target.data.view_as(pred2)).cpu().sum()

            print('Training accuracy: {:.4f} %'.format(float(correct1) / len(target_select_loader.dataset)*100))
            print('Training accuracy: {:.4f} %'.format(float(correct2) / len(target_select_loader.dataset)*100))
                           
            #if domain_flag == 0: # labeled source
            for j in range(pred1.shape[0]):
                spre_label.append([pro1_pre[j], pro2_pre[j]])
                tag_label.append([lab1_pre[j], lab2_pre[j]])
        return spre_label, tag_label


if __name__ == '__main__':
    #'''
    if sel_flag  == 0:
        #for traepo in range(K):
        traepo = 0
        model = models.MDAnetsf(num_classes)
        if args.cuda:
            model.cuda()
        train(traepo, model)
        print('The {} time trainging done!'.format(traepo+1))

    if sel_flag  == 1:
        path='MDA1'
        domain_flag = 1
        if domain_flag == 0:
            for traepo in range(1):
                model = models.MDAnetsf_alex(num_classes)
                if args.cuda:
                    model.cuda()
                #model.load_state_dict(torch.load('{}/{}_{}_MDA_{}_{}{}.pth'.format(modelroot, dataname, args.test_dir, load_tag, load_sel, traepo)))
                model.load_state_dict(torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag)))
                                
                spre_label1, tag_label1, spre_label2, tag_label2 = select(traepo, model, domain_flag)
                del model
                spre_label1 = np.array(spre_label1)
                tag_label1 = np.array(tag_label1)
                spre_label2 = np.array(spre_label2)
                tag_label2 = np.array(tag_label2)
                #if domain_flag == 0:
                if traepo == 0:
                    pre_label1 = tag_label1
                    pre_label1 = np.concatenate((pre_label1,spre_label1),1)
                    pre_label2 = tag_label2
                    pre_label2 = np.concatenate((pre_label2,spre_label2),1)
                else:
                    pre_label1 = np.concatenate((pre_label1,spre_label1),1)
                    pre_label2 = np.concatenate((pre_label2,spre_label2),1)
            np.savetxt('./MDA1/{}_src_{}_{}.csv'.format(dataname, args.source1_dir, args.test_dir), np.array(pre_label1), fmt='%.6f', delimiter=',')      
                       
            np.savetxt('./MDA1/{}_src_{}_{}.csv'.format(dataname, args.source2_dir, args.test_dir), np.array(pre_label2), fmt='%.6f', delimiter=',')      

        if domain_flag == 1:
            for traepo in range(1):
                model = models.MDAnetsf_alex(num_classes)
                if args.cuda:
                    model.cuda()
                model.load_state_dict(torch.load('{}/{}_{}_{}.pth'.format(modelroot, dataname, args.test_dir, train_tag)))
                                
                spre_label, tag_label = select(traepo, model, domain_flag)
                del model                                                 
                
                spre_label = np.array(spre_label)
                tag_label = np.array(tag_label)
                if traepo == 0:
                    pre_label = tag_label
                    pre_label = np.concatenate((pre_label,spre_label),1)

                else:
                    pre_label = np.concatenate((pre_label,tag_label),1)
                    pre_label = np.concatenate((pre_label,spre_label),1)

            #np.savetxt('./MDA1/{}_tar_{}.csv'.format(dataname, args.test_dir), np.array(pre_label), fmt='%.6f', delimiter=',')     
                 
    
        tag_label = np.array(tag_label)
        spre_label = np.array(spre_label)
        spre_label = spre_label[:,0] + spre_label[:,1]

        idx = [idx for idx in range(len(tag_label)) if len(set(tag_label[idx,:]))==1]
        if len(idx) > 0:
            idex = []
            for cls in range(num_classes):
                idx1 = [idx1 for idx1 in idx if tag_label[idx1,0] == cls]
                if len(idx1) > 0:
                    idx2 = [idx2 for idx2 in idx1 if spre_label[idx2] >= np.median(spre_label[idx1])]
                    idex.extend(idx2)
                       
        org_file = datapath + args.test_dir + 'List.txt'
        file_org = open(org_file, 'r').readlines()
        idxall = [idxall for idxall, item in enumerate(file_org)]
        
        out_file = datapath + args.test_dir + 'tra.txt'#srct
        file = open(out_file,'w')                   
        for i in idex:
            lines = file_org[i]
            line = lines.strip().split(' ')
            new_lines = line[0]
            file.write('%s %s\n' % (new_lines, int(tag_label[i,0])))
        file.close()
        out_file = datapath + args.test_dir + 'tst.txt'#left
        file = open(out_file,'w')
        for i in list(set(idxall) - set(idex)):
            lines = file_org[i]
            line = lines.strip().split(' ')
            #new_lines = line[0]
            #file.write('%s %s\n' % (new_lines, int(tag_label[i,0])))
            file.write(lines)
        file.close()
    
