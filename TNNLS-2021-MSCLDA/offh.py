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
import MDANet3 as models
from torch.utils import model_zoo
import numpy as np
import mmd#_pdist as mmd
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
target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
target_num = len(target_test_loader.dataset)

test_result = []
train_loss = []
test_loss = []
source_weight = []

K = 5 # training times
train_tags = ['mscl']
train_flag = 0
train_tag = train_tags[train_flag]

def train(traepo,model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    #count_flag = 0
    early_stop = 15000

    for i in range(1, args.iter + 1):#i=100
        model.train()#model.train(False)
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (i - 1) / (args.iter)), 0.75)
        if (i - 1) % 100 == 0:
            print("learning rate：", LEARNING_RATE)
        optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnetc3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnets3.parameters(), 'lr': LEARNING_RATE},
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
        if args.cuda:
            source_data1, source_label1 = source_data1.cuda(), source_label1.cuda()
            source_data2, source_label2 = source_data2.cuda(), source_label2.cuda()
            source_data3, source_label3 = source_data3.cuda(), source_label3.cuda()
            target_data = target_data.cuda()
        source_data1, source_label1 = Variable(source_data1), Variable(source_label1)
        source_data2, source_label2 = Variable(source_data2), Variable(source_label2)
        source_data3, source_label3 = Variable(source_data3), Variable(source_label3)        
        target_data = Variable(target_data)
        optimizer.zero_grad()

        domain_loss, class_loss, cls_loss, l1_loss, weight = model(source_data1, source_data2, source_data3, target_data, source_label1, mark=1)
                
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss1 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        ws1 = 1 / weight.item()
        loss1.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tclass_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss1.item(), cls_loss.item(), domain_loss.item(), class_loss.item(), l1_loss.item()))
        
        domain_loss, class_loss, cls_loss, l1_loss, weight = model(source_data1, source_data2, source_data3, target_data, source_label2, mark=2)
                
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss2 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        ws2 = 1 / weight.item()
        loss2.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tclass_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss2.item(), cls_loss.item(), domain_loss.item(), class_loss.item(), l1_loss.item()))
        
        domain_loss, class_loss, cls_loss, l1_loss, weight = model(source_data1, source_data2, source_data3, target_data, source_label3, mark=3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        loss3 = cls_loss + gamma * (domain_loss + class_loss + l1_loss)
        ws3 = 1 / weight.item()
        loss3.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tcls_Loss: {:.6f}\tdomain_Loss: {:.6f}\tclass_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / args.iter, loss3.item(), cls_loss.item(), domain_loss.item(), class_loss.item(), l1_loss.item()))
        
        
        w1 = ws1 / (ws1 + ws2 + ws3)
        w2 = ws2 / (ws1 + ws2 + ws3)
        w3 = ws3 / (ws1 + ws2 + ws3)
        
        if i % args.log_interval == 0:
            train_loss.append([loss1.item(), loss2.item(), w1, w2])
            np.savetxt('./MDA/{}_train_loss_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(train_loss), fmt='%.6f', delimiter=',')
                
        if i % (args.log_interval * 10) == 0:
            t_num, t_accu = test(traepo, model, w1, w2, w3)

            t_correct = t_num[3]
            max_correct = max(t_num)
            if t_correct > correct:
                correct = t_correct
                torch.save(model.state_dict(), '{}/{}_{}_MDA_{}{}.pth'.format(modelroot, dataname, args.test_dir, train_tag, traepo))            
            print( "Target %s max correct:" % args.test_dir, correct, "\n")

            t_num.extend(t_accu)
            test_result.append(t_num)
            np.savetxt('./MDA/{}_test_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(test_result), fmt='%.4f', delimiter=',')
            
        if i > early_stop:
            break
                
def test(traepo, model, w1, w2, w3):#sort(w)
    model.eval()
    t_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct = 0
    correctm = 0
    correctw = 0

    
    for data, target in target_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        pred1, pred2, pred3 = model(data)

        pred1 = torch.nn.functional.softmax(pred1, dim=1)
        pred2 = torch.nn.functional.softmax(pred2, dim=1)
        pred3 = torch.nn.functional.softmax(pred3, dim=1)
        preds1 = pred1.data.max(1)[1]
        preds2 = pred2.data.max(1)[1]
        preds3 = pred3.data.max(1)[1]
        
        p13 = preds1.eq(preds3.data.view_as(preds1))
        p23 = preds2.eq(preds3.data.view_as(preds2))#.cpu().sum()
        for index in range(len(p13.data)):
            if p13.data[index] != 1:
                p13.data[index] = 13
        for index in range(len(p13.data)):
            if p23.data[index] != 1:
                p13.data[index] = 23
        coml = p13.eq(p23.data.view_as(p13)).cpu().sum()
        
        w=np.array([w1,w2,w3]) 
        ws = w
        idx = w.argsort()
        a = 20
        b = a / (3*preds1.shape[0])
        c = float(coml) / preds1.shape[0]
        if float(coml) >= a: 
            wmax = 1/(1 + math.exp(-(w[idx[2]] + (1-c)))) 
            wmin = 1/(1 + math.exp(-(w[idx[0]] - (1-c))))
            wmed = 1/(1 + math.exp(-(w[idx[1]])))
            ws[idx[2]] = wmax/(wmax+wmed+wmin) + b + b/2
            ws[idx[0]] = wmin/(wmax+wmed+wmin) - b 
            ws[idx[1]] = wmed/(wmax+wmed+wmin) - b/2
            if ws[idx[0]] < 0:
                ws[idx[1]] = ws[idx[1]] + ws[idx[0]]
                ws[idx[0]] = 0

        source_weight.append([w1, w2, w3, ws[0].item(),ws[1].item(), ws[2].item(), float(coml)])
        np.savetxt('./MDA/{}_source_weight_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(source_weight), fmt='%.6f', delimiter=',')      
       
        pred = pred1.data.max(1)[1]
        correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred2.data.max(1)[1]
        correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = pred3.data.max(1)[1]
        correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = w1*pred1 + w2*pred2 + w3*pred3
        pred = pred.data.max(1)[1]
        correctw += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = (pred1 + pred2 + pred3)/3
        pred = pred.data.max(1)[1]
        correctm += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        pred = ws[0]*pred1 + ws[1]*pred2 + ws[2]*pred3

        t_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
        pred = pred.data.max(1)[1]       
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()


    t_loss /= len(target_test_loader.dataset)
    test_loss.append([t_loss])
    
    np.savetxt('./MDA/{}_test_loss_{}_{}{}.csv'.format(dataname, args.test_dir, train_tag, traepo), np.array(test_loss), fmt='%.6f', delimiter=',')      
    
    accu1 = float(correct1) / len(target_test_loader.dataset)*100 
    accu2 = float(correct2) / len(target_test_loader.dataset)*100
    accu3 = float(correct3) / len(target_test_loader.dataset)*100 
    accu = float(correct) / len(target_test_loader.dataset)*100 
    accuw = float(correctw) / len(target_test_loader.dataset)*100
    accum = float(correctm) / len(target_test_loader.dataset)*100
    correct_num = [correct1, correct2, correct3, correct, correctw, correctm]
    accu = [accu1, accu2, accu3, accu, accuw, accum]
    print(args.test_dir, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            t_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
    print('\nsource1 {}, source2 {}, source3 {}, weight {} mean {}'.format(correct1, correct2, correct3, correctw, correctm))
          
    return correct_num, accu


if __name__ == '__main__':
    #'''
    traepo = 0
    model = models.MDAnet(num_classes)
    if args.cuda:
        model.cuda()
    train(traepo, model)
    print('The {} time trainging done!'.format(traepo+1))


