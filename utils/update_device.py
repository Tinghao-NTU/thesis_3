import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from sklearn.cluster import KMeans
import numpy as np

import copy
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
def feature_cal(weights1, weights2):
    para_list = []
    for (_, parameters1),(_, parameters2) in zip(weights1.items(),weights2.items()):
        para_list.append(torch.norm((parameters1.view(-1)- parameters2.view(-1)), p = 1,dim = 0).cpu().numpy().item())
#     para_list = torch.cat(para_list)
    return para_list

def cluster_devices(wlist):    
    dist_mx = []
    class_index = {i: np.array([], dtype='int64') for i in range(10)}
    
    for i in range(100):
        temp = copy.deepcopy(wlist[i])
        para_list1 = temp['fc2.bias']
        dist_mx.append(para_list1.view(1,-1))
    dist_mx = torch.cat(dist_mx,0)
    
    kmeans = KMeans(n_clusters=10, random_state=0).fit(dist_mx.cpu())
    for i in range(10):
        class_index[i] = np.where(kmeans.labels_ == i)[0]
    return class_index


def local_update(args,dataset_train,dataset_test,dict_users,idxs_users,epoch,net_glob,next_state_list = None,acc_list = None,w_list = None):
    w_locals_temp = []
    loss_locals = []

    if next_state_list is None:
        acc_list = []
        next_state_list = []
        w_list = []
    for idx in idxs_users:
        idx_str = '%d'%(idx)
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx_str])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        result = w['fc2.bias'].view(1,-1)
        w_locals_temp.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        if epoch == 0:
            w_list.append(w)
            next_state_list.append(result)
        else:
            w_list[idx] = copy.deepcopy(w)
            next_state_list[idx] =w['fc2.bias'].view(1,-1)
    if epoch == 0:
        next_state = torch.cat(next_state_list,0)
    w_glob = FedAvg(w_locals_temp)
    
    reward_n = [torch.norm((w_locals_temp[i]['fc2.bias']- w_glob['fc2.bias']).cpu().detach(), p = 1,dim = 0).numpy().item() for i in range(10)]
    reward_n = [1/reward_n[i] for i in range(10)]
    net_glob.load_state_dict(w_glob)
    if epoch == 0:
        next_state = torch.cat((next_state,w_glob['fc2.bias'].view(1,-1)),0)
    else:
        next_state_list[-1] = w_glob['fc2.bias']
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
#         acc_test, loss_test = test_img(net_glob, dataset_test, args)     
    acc_train, loss_train = test_img(net_glob, dataset_test, args)
    acc_list.append(acc_train)
    print('Round {:3d}, Average loss {:.3f}, Training accuracy {:.3f}'.format(epoch, loss_avg,acc_train))
    file_handle=open('TrainingRecords_PG_fc_cifar.txt',mode='a')
    file_handle.write("Round:%.03f; Average loss: %.03f; Training accuracy:%.03f \n"%(epoch,loss_avg,acc_train))
    file_handle.close()
    if acc_train < 55:
        done_n = [0] * 10
    else:
        done_n = [1] * 10
    if epoch == 0 :
        clusteded_dic = cluster_devices(w_list)
        return next_state,next_state_list,net_glob, acc_list,clusteded_dic,w_list
    else:
        return next_state_list,net_glob, acc_list,w_list, reward_n,done_n
    
def get_state(device_idx,state):
    return torch.cat(((state[device_idx],state[-1].view(1,-1))),0).view(-1).cpu().detach().numpy()