import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import json
import copy
import random
import umap
from sklearn import manifold
from utils.sampling import fashion_noniid,mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from sklearn.model_selection import train_test_split
import torch.optim as optim
import gym
from sklearn import preprocessing
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
#from sklearn import datasets, cluster
from utils.update_device_MLP import local_update,get_state,permute_device,feature_cal
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,weigth_init,CNNMnist_Compare,CNNCifar_Compare
from models.Fed import FedAvg,FedAvg_weighted
from models.test import test_img

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

args.dataset = 'cifar'
args.local_ep = 5
args.local_bs = 20
args.model = 'cnn'
args.num_users = 100
args.servers = 5
args.iid = False

if args.dataset == 'fashion':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.FashionMNIST(
        '../data/fashion', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))
    dataset_test = datasets.FashionMNIST(
            '../data/fashion', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))

    dict_users,pre = fashion_noniid(dataset_train, args.num_users,0.8,None)

# load dataset and split users
if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users,pre= mnist_noniid(dataset_train, args.num_users,0.8,None)

elif args.dataset == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users,pre= cifar_noniid(dataset_train, args.num_users,0.8,None)
img_size = dataset_train[0][0].shape

# build model
if args.model == 'cnn' and args.dataset == 'cifar':
    net_glob = CNNCifar(args=args).to(args.device)
elif args.model == 'cnn' and args.dataset == 'mnist':
    net_glob = CNNMnist(args=args).to(args.device)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
else:
    exit('Error: unrecognized model')
print(net_glob)
net_glob.train()
net_glob.apply(weigth_init)
# copy weights
w_glob = net_glob.state_dict()
with open('data_cifar.json', 'r') as f:
    dict_users = json.load(f)

with open('device_index_cifar.json', 'r') as f:
    clusteded_dic = json.load(f)


class_index = [[] for i in range(10)]
for i in range(10):
    class_index[i] = np.where(np.array(pre) == i)[0].tolist()

# training
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
np.random.seed(0)

n_episodes= 0
iterations_per_episode = 50
eps_start=1.0
eps_end = 0.01
eps_decay=0.99
verbose = False
edge_iter = 5
all_rewards = [-10000000]
length_list = [1000000]
scores = []
preset_accuracy = torch.tensor([[87.0]])
balance = True
name_acc = 'Records_noniid_cnn_cifar_bala.txt'

def alloc_devices(sele_devices,edge_num,index_edge_num):
    index_edge_users = []
    for i in range(edge_num):
        index_edge_users.append(sele_devices[0:index_edge_num[i]])
        sele_devices =  np.delete(sele_devices,np.arange(index_edge_num[i]))
    return index_edge_users

def cluster_select(clusteded_dic,sele_num,epoch,pool = None):
    class_num = len(clusteded_dic)
    results = []
    total_num = sele_num * 10
    unsele_num = 0
    len_list = np.array([len(clusteded_dic[str(i)]) for i in range(10)])
    sufficient_list = np.where(len_list > sele_num)[0]
    every_device = np.arange(args.num_users)
    if epoch == 0:
        for i in range(10):
            if len(clusteded_dic[str(i)]) >= sele_num:
                sele_devices = np.random.choice(clusteded_dic[str(i)],sele_num,replace=False).tolist()
            else:
                sele_devices = clusteded_dic[str(i)]
                unsele_num += sele_num - len(clusteded_dic[str(i)])
            results.extend(sele_devices)
            if pool is not None:
                pool[i] = sele_devices
        remain_devices = np.setdiff1d(every_device, np.array(results)).tolist()
        results.extend(np.random.choice(remain_devices,unsele_num,replace=False).tolist())
        
        return np.random.permutation(results),pool

    else:
        for i in range(class_num):
            all_devices = clusteded_dic[str(i)]
            selected_devices = pool[i]
            overlap_devices = np.intersect1d(all_devices, selected_devices).tolist()
            remain_devices = np.setdiff1d(all_devices, selected_devices).tolist()
            if len(clusteded_dic[str(i)]) < sele_num:
                results.extend(clusteded_dic[str(i)])
            elif len(remain_devices) - sele_num < 0:
                sele_list = remain_devices
#                 print(sele_list)
                reused_devices = np.random.choice(overlap_devices,sele_num - len(remain_devices) ,replace=False).tolist()
                pool[i] = reused_devices
                results.extend(reused_devices + sele_list)
            else:
                sele_list = np.random.choice(remain_devices,sele_num,replace=False).tolist()
                pool[i].extend(sele_list)
                results.extend(sele_list)
                
        
        remain_devices = np.setdiff1d(every_device, np.array(results)).tolist()
        results.extend(np.random.choice(remain_devices,total_num-len(results),replace=False).tolist())    
            
        return np.random.permutation(results),pool

eps = eps_start
max_score=-20-1
info = '100_'
for ep_iter in range(0, n_episodes+1):
    score = 0
    policy_loss_total = 0
    w_list = []
    policy_reward = []
    length_temp = 0
    epoch = 0
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)#CNNCifar_Compare().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist_Compare().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.apply(weigth_init)
    print(net_glob)
    net_glob.train()
    #torch.save(net_glob.state_dict(), 'parameter_cifar_new.pkl')
    net_glob.load_state_dict(torch.load('parameter_cifar_new.pkl'))
    w_glob = net_glob.state_dict()

    idxs_users = [i for i in range(args.num_users)]
    np.random.seed(0)
    idxs_users_temp = np.random.permutation(idxs_users)

    index_edge_num = []
    sele_num = int(args.num_users * 1)
    sele_each_num = int(sele_num/10)
    for i in range(args.servers):
        if i == args.servers-1:
            index_edge_num.append(int(sele_num- (args.servers-1)*int(sele_num/args.servers)))
        else:
            index_edge_num.append(int(sele_num/args.servers))

    net_glob_init = [] 
    for i in range(args.servers):
        net_glob_init.append(copy.deepcopy(net_glob))    

    acc_list_init = [[] for i in range(args.servers)]                              
    w_list = [ [] for i in range(args.num_users)]                                
    state_matrix = [ [] for i in range(args.servers)] 
    done = 0

    net_glob = copy.deepcopy(net_glob_init)
    acc_list = copy.deepcopy(acc_list_init)
    w_list_init = copy.deepcopy(w_list)
    w_glob = [net_glob[i].state_dict() for i in range(args.servers)] 
    epoch = 0
    done = 0    
       
    for roll_num in range(5):   
        index_edge_num = []
        sele_num = 10
        for i in range(args.servers):
            if i == args.servers-1:
                index_edge_num.append(int(sele_num- (args.servers-1)*int(sele_num/args.servers)))
            else:
                index_edge_num.append(int(sele_num/args.servers))
        pool =  [ [] for i in range(10)]
        args.lr = 0.01
        done = 0
        epoch = 1
        net_glob = copy.deepcopy(net_glob_init)
        acc_list = copy.deepcopy(acc_list_init)
        w_list = copy.deepcopy(w_list_init)
        info = '10_'
        while not done: 
            idxs_users = [i for i in range(args.num_users)]
            sele_devices= np.random.permutation(idxs_users)[:10]
            index_edge_users = alloc_devices(sele_devices,args.servers,index_edge_num)
            print(index_edge_users)
            for j in range(edge_iter):
                for i in range(args.servers):
                    state_matrix_temp,net_glob_temp,acc_list_temp,_,w_list= local_update(args,dataset_train,
                                                                dataset_test,dict_users,index_edge_users[i],epoch,net_glob[i],acc_list = acc_list[i],preset = preset_accuracy,w_list =w_list)
                    state_matrix[i]=state_matrix_temp
                    net_glob[i]= net_glob_temp
                    acc_list[i].append(acc_list_temp)


            w_glob_list = [net_glob[j].state_dict() for j in range(args.servers)]
            w_glob = FedAvg_weighted(w_glob_list,index_edge_num)
            for i in range(args.servers):
                net_glob[i].load_state_dict(w_glob)        
            acc_train, loss_train = test_img(net_glob[0], dataset_test, args)
            file_handle=open('fedavg_'+info+args.dataset+'_'+str(roll_num)+'.txt',mode='a')
            file_handle.write(str(acc_train.numpy().item()))
            file_handle.write('\n')
            file_handle.close()
            args.lr = args.lr * 0.99

            if epoch >= 100:
                done_n = [1] * 150
                done = 1

            epoch += 1      
     
    for roll_num in range(5):   
        index_edge_num = []
        sele_num = 30
        for i in range(args.servers):
            if i == args.servers-1:
                index_edge_num.append(int(sele_num- (args.servers-1)*int(sele_num/args.servers)))
            else:
                index_edge_num.append(int(sele_num/args.servers))
        pool =  [ [] for i in range(10)]
        args.lr = 0.01
        info = '30_'
        done = 0
        epoch = 1
        net_glob = copy.deepcopy(net_glob_init)
        acc_list = copy.deepcopy(acc_list_init)
        w_list = copy.deepcopy(w_list_init)

        while not done: 
            idxs_users = [i for i in range(args.num_users)]
            sele_devices= np.random.permutation(idxs_users)[:30]
            index_edge_users = alloc_devices(sele_devices,args.servers,index_edge_num)
            print(index_edge_users)
            for j in range(edge_iter):
                for i in range(args.servers):
                    state_matrix_temp,net_glob_temp,acc_list_temp,_,w_list= local_update(args,dataset_train,
                                                                dataset_test,dict_users,index_edge_users[i],epoch,net_glob[i],acc_list = acc_list[i],preset = preset_accuracy,w_list =w_list)
                    state_matrix[i]=state_matrix_temp
                    net_glob[i]= net_glob_temp
                    acc_list[i].append(acc_list_temp)


            w_glob_list = [net_glob[j].state_dict() for j in range(args.servers)]
            w_glob = FedAvg_weighted(w_glob_list,index_edge_num)
            for i in range(args.servers):
                net_glob[i].load_state_dict(w_glob)        
            acc_train, loss_train = test_img(net_glob[0], dataset_test, args)
            file_handle=open('fedavg_'+info+args.dataset+'_'+str(roll_num)+'.txt',mode='a')
            file_handle.write(str(acc_train.numpy().item()))
            file_handle.write('\n')
            file_handle.close()
            args.lr = args.lr * 0.99

            if epoch >= 80:
                done_n = [1] * 150
                done = 1

            epoch += 1    
       
    for roll_num in range(5):   
        index_edge_num = []
        sele_num = 50
        for i in range(args.servers):
            if i == args.servers-1:
                index_edge_num.append(int(sele_num- (args.servers-1)*int(sele_num/args.servers)))
            else:
                index_edge_num.append(int(sele_num/args.servers))
        pool =  [ [] for i in range(10)]
        args.lr = 0.01
        done = 0
        info = '50_'
        epoch = 1
        net_glob = copy.deepcopy(net_glob_init)
        acc_list = copy.deepcopy(acc_list_init)
        w_list = copy.deepcopy(w_list_init)

        while not done: 
            idxs_users = [i for i in range(args.num_users)]
            sele_devices= np.random.permutation(idxs_users)[:50]
            index_edge_users = alloc_devices(sele_devices,args.servers,index_edge_num)
            print(index_edge_users)
            for j in range(edge_iter):
                for i in range(args.servers):
                    state_matrix_temp,net_glob_temp,acc_list_temp,_,w_list= local_update(args,dataset_train,
                                                                dataset_test,dict_users,index_edge_users[i],epoch,net_glob[i],acc_list = acc_list[i],preset = preset_accuracy,w_list =w_list)
                    state_matrix[i]=state_matrix_temp
                    net_glob[i]= net_glob_temp
                    acc_list[i].append(acc_list_temp)


            w_glob_list = [net_glob[j].state_dict() for j in range(args.servers)]
            w_glob = FedAvg_weighted(w_glob_list,index_edge_num)
            for i in range(args.servers):
                net_glob[i].load_state_dict(w_glob)        
            acc_train, loss_train = test_img(net_glob[0], dataset_test, args)
            file_handle=open('fedavg_'+info+args.dataset+'_'+str(roll_num)+'.txt',mode='a')
            file_handle.write(str(acc_train.numpy().item()))
            file_handle.write('\n')
            file_handle.close()
            args.lr = args.lr * 0.99

            if epoch >= 80:
                done_n = [1] * 150
                done = 1

            epoch += 1                       
        
    for roll_num in range(5):   
        index_edge_num = []
        sele_num = 70
        for i in range(args.servers):
            if i == args.servers-1:
                index_edge_num.append(int(sele_num- (args.servers-1)*int(sele_num/args.servers)))
            else:
                index_edge_num.append(int(sele_num/args.servers))
        pool =  [ [] for i in range(10)]
        info = '70_'
        args.lr = 0.01
        done = 0
        epoch = 1
        net_glob = copy.deepcopy(net_glob_init)
        acc_list = copy.deepcopy(acc_list_init)
        w_list = copy.deepcopy(w_list_init)

        while not done: 
            idxs_users = [i for i in range(args.num_users)]
            sele_devices= np.random.permutation(idxs_users)[:70]
            index_edge_users = alloc_devices(sele_devices,args.servers,index_edge_num)
            print(index_edge_users)
            for j in range(edge_iter):
                for i in range(args.servers):
                    state_matrix_temp,net_glob_temp,acc_list_temp,_,w_list= local_update(args,dataset_train,
                                                                dataset_test,dict_users,index_edge_users[i],epoch,net_glob[i],acc_list = acc_list[i],preset = preset_accuracy,w_list =w_list)
                    state_matrix[i]=state_matrix_temp
                    net_glob[i]= net_glob_temp
                    acc_list[i].append(acc_list_temp)


            w_glob_list = [net_glob[j].state_dict() for j in range(args.servers)]
            w_glob = FedAvg_weighted(w_glob_list,index_edge_num)
            for i in range(args.servers):
                net_glob[i].load_state_dict(w_glob)        
            acc_train, loss_train = test_img(net_glob[0], dataset_test, args)
            file_handle=open('fedavg_'+info+args.dataset+'_'+str(roll_num)+'.txt',mode='a')
            file_handle.write(str(acc_train.numpy().item()))
            file_handle.write('\n')
            file_handle.close()
            args.lr = args.lr * 0.99

            if epoch >= 80:
                done_n = [1] * 150
                done = 1

            epoch += 1           