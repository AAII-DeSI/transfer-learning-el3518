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
import DCANet as models
from torch.utils import model_zoo
import numpy as np
import mmd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
model_num = 0

classpath = datapath + domains[0] + '/'
classlist = os.listdir(classpath)
classlist.sort()

sam_flag = 0
source_sam = ['pre']
source_sel = source_sam[sam_flag]
list_sam = ['List']
sel_sam = list_sam[sam_flag]
load_sel = source_sam[0]

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
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

source1_loader = data_loader.load_training(args.root_path, args.source1_dir, args.batch_size, kwargs)
source2_loader = data_loader.load_training(args.root_path, args.source2_dir, args.batch_size, kwargs)
source3_loader = data_loader.load_training(args.root_path, args.source3_dir, args.batch_size, kwargs)
target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
target_num = len(target_test_loader.dataset)

test_result = []
train_loss = []
test_loss = []
source_weight = []

K = 5 # training times
train_tags = ['dca']
train_flag = 0
train_tag = train_tags[train_flag]
load_tag = train_tags[train_flag]
pse_num = 2000
def train(traepo,model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    #target_iter = iter(target_train_loader)   
		
    correct = 0
    early_stop = 5000

    for i in range(1, args.iter + 1):
        model.train()
        if i>pse_num:
            target_train_loader = data_loader.load_image_TSS(args.root_path, args.test_dir, classlist, 'left', args.batch_size, kwargs)
            target_iter = iter(target_train_loader)
            sourcet_loader = data_loader.load_image_TSS(args.root_path, args.test_dir, classlist, 'tra', args.batch_size, kwargs)
            sourcet_iter = iter(sourcet_loader)
            LEARNING_RATE = args.lr / math.pow((1 + 10 * ((i-pse_num) - 1) / (args.iter)), 0.75)
        if i<=pse_num:
            target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
            target_iter = iter(target_train_loader)
            sourcet_loader = source1_loader
            sourcet_iter = iter(source1_loader)
            LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        #LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.member1.parameters()},
            {'params': model.member2.parameters()},
            {'params': model.member3.parameters()},
            {'params': model.mweight1.parameters()},
            {'params': model.mweight2.parameters()},
            {'params': model.mweight3.parameters()},
            {'params': model.domain.parameters()},
            {'params': model.cls_fc_son11.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son12.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son13.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son21.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son22.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son23.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son31.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son32.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son33.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
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
        try:
            source_data3, source_label3 = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data3, source_label3 = source3_iter.next() 
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        try:
            source_datat, source_labelt = sourcet_iter.next()
        except Exception as err:
            sourcet_iter = iter(sourcet_loader)
            source_datat, source_labelt = sourcet_iter.next()
        
        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            source_data3, source_label3 = source_data3.cuda(), source_label3.cuda()
            source_datat, source_labelt = source_datat.cuda(), source_labelt.cuda()
            target_data = target_data.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        source_data3, source_label3 = Variable(source_data3), Variable(source_label3)
        source_datat, source_labelt = Variable(source_datat), Variable(source_labelt)         
        target_data = Variable(target_data)
        optimizer.zero_grad()
        
        domain_loss, cls_loss, l1_loss, clsdomain_loss = model(source_data1, source_data2, source_data3, source_datat, 
                                                               target_data, source_label1, source_labelt, i, pse_num, mark=1)
        
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss1 = cls_loss + gamma * (domain_loss +  l1_loss + clsdomain_loss)# 
        loss1.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tl1_Loss: {:.6f}\tclsdomain_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss1.item(), cls_loss.item(), domain_loss.item(), l1_loss.item(), clsdomain_loss.item()))
        
        domain_loss, cls_loss, l1_loss, clsdomain_loss = model(source_data1, source_data2, source_data3, source_datat,
                                                               target_data, source_label2, source_labelt, i, pse_num, mark=2)
        
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss2 = cls_loss + gamma * (domain_loss + l1_loss + clsdomain_loss)#  
        loss2.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tl1_Loss: {:.6f}\tclsdomain_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss2.item(), cls_loss.item(), domain_loss.item(), l1_loss.item(), clsdomain_loss.item()))        
        
               
        domain_loss, cls_loss, l1_loss, clsdomain_loss = model(source_data1, source_data2, source_data3, source_datat,
                                                               target_data, source_label3, source_labelt, i, pse_num, mark=3)
        
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss3 = cls_loss + gamma * (domain_loss + l1_loss + clsdomain_loss)#  
        loss3.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tl1_Loss: {:.6f}\tclsdomain_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss3.item(), cls_loss.item(), domain_loss.item(), l1_loss.item(), clsdomain_loss.item()))        
        
        if i % args.log_interval == 0:
            train_loss.append([loss1.item(), loss2.item(), loss3.item()])
            np.savetxt('./MDA/{}_train_loss_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(train_loss), fmt='%.6f', delimiter=',')
         
        
        if i % (args.log_interval * 10) == 0:
            t_num, t_accu, num1, num2, num3 = test(traepo, model)            

            t_correct = t_num[3]
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_MDA_{}{}.pth'.format(modelroot, dataname, args.test_dir, train_tag, traepo))            
            print( "Target %s max correct:" % args.test_dir, correct, "\n")

            t_num.extend(t_accu)
            t_num.extend(num1)
            t_num.extend(num2)
            t_num.extend(num3)
            test_result.append(t_num)
            np.savetxt('./MDA/{}_test_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(test_result), fmt='%.4f', delimiter=',')
            
        if (i-pse_num) % (args.log_interval * 100) == 0 or i == pse_num:
            pselabt(traepo, model, i)
        if i > early_stop:
            break
                
def test(traepo, model):
    model.eval()
    t_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct = 0
    correctm = 0
    
    correct11 = 0
    correct12 = 0
    correct13 = 0
    
    correct21 = 0
    correct22 = 0
    correct23 = 0
    
    correct31 = 0
    correct32 = 0
    correct33 = 0
    
    
    for data, target in target_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred1, pred2, pred3, preda, s1, s2, s3 = model(data)

        pred1 = torch.nn.functional.softmax(pred1, dim=1)
        pred2 = torch.nn.functional.softmax(pred2, dim=1)
        pred3 = torch.nn.functional.softmax(pred3, dim=1)
        preda = torch.nn.functional.softmax(preda, dim=1)                
    
        pred = pred1.data.max(1)[1]
        correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred3.data.max(1)[1]
        correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()
                
        pred = (pred1 + pred2 + pred3)/3
        pred = pred.data.max(1)[1]
        correctm += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        t_loss += F.nll_loss(F.log_softmax(preda, dim=1), target).item()
        pred = preda.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() 
        
        pred1 = torch.nn.functional.softmax(s1[0], dim=1)
        pred2 = torch.nn.functional.softmax(s1[1], dim=1)
        pred3 = torch.nn.functional.softmax(s1[2], dim=1)
        
        pred = pred1.data.max(1)[1]
        correct11 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct12 += pred.eq(target.data.view_as(pred)).cpu().sum()
                
        pred = pred3.data.max(1)[1]
        correct13 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred1 = torch.nn.functional.softmax(s2[0], dim=1)
        pred2 = torch.nn.functional.softmax(s2[1], dim=1)
        pred3 = torch.nn.functional.softmax(s2[2], dim=1)
        
        pred = pred1.data.max(1)[1]
        correct21 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct22 += pred.eq(target.data.view_as(pred)).cpu().sum()
                
        pred = pred3.data.max(1)[1]
        correct23 += pred.eq(target.data.view_as(pred)).cpu().sum()   
        
        pred1 = torch.nn.functional.softmax(s3[0], dim=1)
        pred2 = torch.nn.functional.softmax(s3[1], dim=1)
        pred3 = torch.nn.functional.softmax(s3[2], dim=1)
        
        pred = pred1.data.max(1)[1]
        correct31 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct32 += pred.eq(target.data.view_as(pred)).cpu().sum()
                
        pred = pred3.data.max(1)[1]
        correct33 += pred.eq(target.data.view_as(pred)).cpu().sum()       

    t_loss /= len(target_test_loader.dataset)
    test_loss.append([t_loss])
        
    accu1 = float(correct1) / len(target_test_loader.dataset)*100 
    accu2 = float(correct2) / len(target_test_loader.dataset)*100 
    accu3 = float(correct3) / len(target_test_loader.dataset)*100 
    accu = float(correct) / len(target_test_loader.dataset)*100 
    accum = float(correctm) / len(target_test_loader.dataset)*100 
    accu11 = float(correct11) / len(target_test_loader.dataset)*100 
    accu12 = float(correct12) / len(target_test_loader.dataset)*100 
    accu13 = float(correct13) / len(target_test_loader.dataset)*100 
    accu21 = float(correct21) / len(target_test_loader.dataset)*100 
    accu22 = float(correct22) / len(target_test_loader.dataset)*100 
    accu23 = float(correct23) / len(target_test_loader.dataset)*100 
    accu31 = float(correct31) / len(target_test_loader.dataset)*100 
    accu32 = float(correct32) / len(target_test_loader.dataset)*100 
    accu33 = float(correct33) / len(target_test_loader.dataset)*100 
    correct_num = [correct1, correct2, correct3, correct, correctm]   
    accu = [accu1, accu2, accu3, accu, accum] 
    num1 = [correct11, correct12, correct13, accu11, accu12, accu13]
    num2 = [correct21, correct22, correct23, accu21, accu22, accu23]
    num3 = [correct31, correct32, correct33, accu31, accu32, accu33]
    print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            t_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
    print('\nsource1 {}, source2 {}, source3 {}'.format(correct1, correct2, correct3))
          
    return correct_num, accu, num1, num2, num3


def pselabt(traepo, model, epo):
    if epo > pse_num:
        target_select_loader = data_loader.load_image_TSS_select(args.root_path, args.test_dir, classlist, 'left', args.batch_size, kwargs)
        model.eval()
        spre_label = []
        tag_label = []

        for data, target in target_select_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
                
            p1, p2, p3, p, s1, s2, s3 = model(data)        

            pred11 = torch.nn.functional.softmax(s1[0], dim=1)
            pred12 = torch.nn.functional.softmax(s1[1], dim=1)
            pred13 = torch.nn.functional.softmax(s1[2], dim=1)
            pred21 = torch.nn.functional.softmax(s2[0], dim=1)
            pred22 = torch.nn.functional.softmax(s2[1], dim=1)
            pred23 = torch.nn.functional.softmax(s2[2], dim=1)
            pred31 = torch.nn.functional.softmax(s3[0], dim=1)
            pred32 = torch.nn.functional.softmax(s3[1], dim=1)
            pred33 = torch.nn.functional.softmax(s3[2], dim=1)
            pred = (pred11 + pred12 +pred13 + pred21 + pred22 + pred23 + pred31 + pred32 + pred33)/9
                
            pro = np.array(pred.data.cpu())
            pro_pre = np.array(pred.data.max(1)[0].cpu())
            
            lab11_pre = np.array(pred11.data.max(1)[1].cpu())
            lab12_pre = np.array(pred12.data.max(1)[1].cpu()) 
            lab13_pre = np.array(pred13.data.max(1)[1].cpu())
            lab21_pre = np.array(pred21.data.max(1)[1].cpu())
            lab22_pre = np.array(pred22.data.max(1)[1].cpu())   
            lab23_pre = np.array(pred23.data.max(1)[1].cpu()) 
            lab31_pre = np.array(pred31.data.max(1)[1].cpu())
            lab32_pre = np.array(pred32.data.max(1)[1].cpu())   
            lab33_pre = np.array(pred33.data.max(1)[1].cpu())        

            for j in range(pred.shape[0]):
                spre_label.append([pro_pre[j]])
                tag_label.append([lab11_pre[j], lab12_pre[j], lab13_pre[j], lab21_pre[j], lab22_pre[j], lab23_pre[j], lab31_pre[j], lab32_pre[j], lab33_pre[j]])
        
        tag_label = np.array(tag_label)
        spre_label = np.array(spre_label)

        idx = [idx for idx in range(len(tag_label)) if len(set(tag_label[idx,:]))==1]
        if len(idx) > 0:
            idex = []
            for cls in range(num_classes):
                idx1 = [idx1 for idx1 in idx if tag_label[idx1,0] == cls]
                if len(idx1) > 0:
                    idx_sort = np.argsort(-spre_label[idx1])
                    idx_tar = np.array(idx1)[np.array(idx_sort)[0]]
                    idex.extend(idx_tar)
            
            org_file = datapath + args.test_dir + 'left.txt'
            file_org = open(org_file, 'r').readlines()
            idxall = [idxall for idxall, item in enumerate(file_org)]
            
            out_file = datapath + args.test_dir + 'tra.txt'
            file = open(out_file,'a')                   
            for i in idex:
                lines = file_org[i]
                line = lines.strip().split(' ')
                new_lines = line[0]
                file.write('%s %s\n' % (new_lines, int(tag_label[i,0])))
            file.close()
            out_file = datapath + args.test_dir +'left.txt'
            file = open(out_file,'w')
            for i in list(set(idxall) - set(idex)):
                lines = file_org[i]
                file.write(lines)
            file.close()
    if epo == pse_num:
        target_select_loader = data_loader.load_image_TSS_select(args.root_path, args.test_dir, classlist, 'List', args.batch_size, kwargs)
        model.eval()
        spre_label = []
        tag_label = []
        
        for data, target in target_select_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
                
            p1, p2, p3, p, s1, s2, s3 = model(data)        

            pred11 = torch.nn.functional.softmax(s1[0], dim=1)
            pred12 = torch.nn.functional.softmax(s1[1], dim=1)
            pred13 = torch.nn.functional.softmax(s1[2], dim=1)
            pred21 = torch.nn.functional.softmax(s2[0], dim=1)
            pred22 = torch.nn.functional.softmax(s2[1], dim=1)
            pred23 = torch.nn.functional.softmax(s2[2], dim=1)
            pred31 = torch.nn.functional.softmax(s3[0], dim=1)
            pred32 = torch.nn.functional.softmax(s3[1], dim=1)
            pred33 = torch.nn.functional.softmax(s3[2], dim=1)
            pred = (pred11 + pred12 +pred13 + pred21 + pred22 + pred23 + pred31 + pred32 + pred33)/9
                
            pro = np.array(pred.data.cpu())
            pro_pre = np.array(pred.data.max(1)[0].cpu())
            
            lab11_pre = np.array(pred11.data.max(1)[1].cpu())
            lab12_pre = np.array(pred12.data.max(1)[1].cpu()) 
            lab13_pre = np.array(pred13.data.max(1)[1].cpu())
            lab21_pre = np.array(pred21.data.max(1)[1].cpu())
            lab22_pre = np.array(pred22.data.max(1)[1].cpu())   
            lab23_pre = np.array(pred23.data.max(1)[1].cpu()) 
            lab31_pre = np.array(pred31.data.max(1)[1].cpu())
            lab32_pre = np.array(pred32.data.max(1)[1].cpu())   
            lab33_pre = np.array(pred33.data.max(1)[1].cpu())        

            for j in range(pred.shape[0]):
                spre_label.append([pro_pre[j]])
                tag_label.append([lab11_pre[j], lab12_pre[j], lab13_pre[j], lab21_pre[j], lab22_pre[j], lab23_pre[j], lab31_pre[j], lab32_pre[j], lab33_pre[j]])
        
        tag_label = np.array(tag_label)
        spre_label = np.array(spre_label)
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
            
            out_file = datapath + args.test_dir + 'tra.txt'
            file = open(out_file,'a')                   
            for i in idex:
                lines = file_org[i]
                line = lines.strip().split(' ')
                new_lines = line[0]
                file.write('%s %s\n' % (new_lines, int(tag_label[i,0])))
            file.close()
            out_file = datapath + args.test_dir +'left.txt'
            file = open(out_file,'w')
            for i in list(set(idxall) - set(idex)):
                lines = file_org[i]
                file.write(lines)
            file.close()
        

if __name__ == '__main__':
    
    for traepo in range(K):
        model = models.DCAnet3(num_classes)
        if args.cuda:
            model.cuda()
        train(traepo, model)
        print('The {} time trainging done!'.format(traepo+1))
        os.remove(datapath + args.test_dir + 'left.txt')
        os.remove(datapath + args.test_dir + 'tra.txt')
