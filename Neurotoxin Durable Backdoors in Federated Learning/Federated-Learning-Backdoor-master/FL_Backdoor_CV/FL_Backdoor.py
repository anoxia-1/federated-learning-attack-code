import os
import numpy as np
import time
import argparse
import sys
from math import ceil
from random import Random
import time
import random
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim_tr
from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import datetime
from scipy.io import loadmat
import json
from scipy import io
from torch.optim.lr_scheduler import LambdaLR
import math
import scipy.misc
import copy
from comm_helpers import SyncAllreduce
import utils_v1 as util
import LocalSGD as optim
from universal_pert import universal_perturbation
import scipy.io as scio
import sys
import os
from utils_v1 import *
import LocalSGD as optim
import torch.optim as optim_torch
sys.stdout.flush()

parser = argparse.ArgumentParser(description='FL Backdoor CIFAR-10 baseline')

parser.add_argument('--experiment_name',
                    default="Cifar10_FL_Backdoor",
                    type=str,
                    help='name of this experiment')

parser.add_argument('--ip_address',
                    default="10.129.2.142",
                    type=str,
                    help='ip_address')
parser.add_argument('--GPU_list',
                    default='0',
                    type=str,
                    help='gpu list')
parser.add_argument('--dataset',
                    default="cifar10",
                    type=str,
                    help='dataset name')
parser.add_argument('--model',
                    default="res",
                    type=str,
                    help='neural network model')
parser.add_argument('--rank',
                    default=0,
                    type=int,
                    help='the rank of worker')
parser.add_argument('--size',
                    default=50,
                    type=int,
                    help='size of the pool of participants')
parser.add_argument('--num_comm_ue',
                    default=10,
                    type=int,
                    help='number of participants drawn from the pool at each round')

parser.add_argument('--NIID',
                    default=1,
                    type=int,
                    help='NIID is 1 means the non-iid setting, else NIID is 0')
parser.add_argument('--epoch',
                    default=300,
                    type=int,
                    help='total number of communication rounds')
parser.add_argument('--lr',
                    default=0.16,
                    type=float,
                    help='learning rate')
parser.add_argument('--cp',
                    default=8,
                    type=int,
                    help='communication period / work per clock')
parser.add_argument('--iteration',
                    default=16+1,
                    type=int,
                    help='number of batches per communication round')
parser.add_argument('--bs',
                    default=64,
                    type=int,
                    help='batch size on each worker')
parser.add_argument('--warmup_epoch', default=5, type=int,
                    help='whether to warmup learning rate for first 5 epochs')
'''
nargs='+' 表示 --schedule 后面可以有多个值。这里的 + 表示可以接受一个或多个值，并将这些值存储在一个列表中。如果用户在命令行中提供了多个值，这些值将被解析为一个列表。
'''
parser.add_argument('--schedule', nargs='+', default=None,
                    type=float, help='learning rate schedule')
parser.add_argument('--num_expand_x', default=5000, type=int,  ### 65536
                    help='number of examples')
#### parameters of Backdoor.
parser.add_argument('--attack_epoch',
                    default=80,
                    type=int,
                    help='the epoch in which the attacker appears')

parser.add_argument('--edge_case',
                    default=0,
                    type=int,
                    help='edge_case==1: use cifar-100-class-3; edge_case==0: use cifar-10-class-3')
parser.add_argument('--attack_type',
                    default="edge_case_low_freq_adver_pattern",
                    type=str,
                    help='attack_type: pattern, semantic, edge_case_adver, edge_case_adver_pattern, edge_case_low_freq_adver, edge_case_low_freq_adver_pattern')
parser.add_argument('--attack_target',
                    default=1,
                    type=int,
                    help='the target class. The attacker wants the classifier to misclassify the data into class ""attack_target" ')
parser.add_argument('--attacker_user_id',
                    default=10,
                    type=int,
                    help='participants with id in range(0, args.attacker_user_id) are all attackers.')
parser.add_argument('--attack_activate_round',
                    default=250,
                    type=int,
                    help='after the round >= attack_activate_round, the attack come into play')
parser.add_argument('--base_image_class',
                    default=3,
                    type=int,
                    help='the class of the base image which will be used as trigger in the future')

parser.add_argument('--one_shot_attack',
                    default=0,
                    type=int,
                    help='one shot attack or not')

#### parameters for denfence
parser.add_argument('--NDC',
                    default=1,
                    type=int,
                    help='norm difference clipping or not')
parser.add_argument('--s_norm',
                    default=200.0,
                    type=float,
                    help='norm difference threshold to clip')

parser.add_argument('--fine_tuning_start_round',
                    default=300,
                    type=int,
                    help='the round that begin fine-tuning')

### model replacement
parser.add_argument('--weights_scale',
                    default=0,
                    type=int,
                    help='model replacement with scaled weights')


parser.add_argument('--clip',
                    default=0.25,
                    type=float,
                    help='clip')#norm_bound

parser.add_argument('--scale_weights',
                    default=100.0,
                    type=float,
                    help='scale_weights')

parser.add_argument('--grad_mask',
                    default=0,
                    type=int,
                    help='grad_mask')

parser.add_argument('--norm_bound',
                    default=0,
                    type=int,
                    help='norm_bound')

parser.add_argument('--s_norm_attack',
                    default=5.0,
                    type=float,
                    help='s_norm_attack')

parser.add_argument('--master_node',
                    type=str,
                    help='master node name/IP address for distributed training.')
parser.add_argument('--master_port',
                    default="29021",
                    type=str,
                    help='master port of the distributed training')

############ run_slurm
parser.add_argument('--run_slurm',
                    default=1,
                    type=int,
                    help='run_slurm')

args = parser.parse_args()
args.lr_schedule = {}
if args.schedule is None:
    args.schedule = [150, 0.1, 250, 0.1] #### If epochs=300, schedule = ?;
i, epoch = 0, None
for v in args.schedule:
    if i == 0:
        epoch = v
    elif i == 1:
        args.lr_schedule[epoch] = v
    i = (i + 1) % 2
del args.schedule

args.poison_sentences = ['pasta from Astoria tastes delicious']
print(args)
'''

这段代码定义了一个名为 LabelSmoothLoss 的自定义损失函数类，用于实现标签平滑（Label Smoothing）损失。标签平滑是一种用于改善模型在分类问题中性能的技术，
它通过在目标标签上引入少量的噪声来减轻模型对于训练数据的过拟合。
'''
class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.9):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = 0.9
    '''
    def forward(self, input, target):: 定义了前向传播方法。该方法接受模型的输出 input 和真实标签 target 作为输入，并返回计算得到的损失。
    log_prob = F.log_softmax(input, dim=-1): 对模型的输出进行 log-softmax 操作，得到对数概率。
    '''
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

'''

这个函数用于生成一个带有预热（warm-up）和余弦学习率调度（cosine learning rate schedule）的优化器（optimizer）。
这种学习率调度在训练深度学习模型时很常见，它可以在训练开始时使用较小的学习率进行预热，然后在余弦函数形状的曲线上进行周期性的调整。
'''
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))

        num_cycles = 7.0/16.0
        return max(0.00000001, math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def Get_Model(args):
    model = util.select_model(args.model, args).cuda()
    return model

def Get_LabelSmoothLoss(args):
    criterion_labelsmoothLoss = LabelSmoothLoss().cuda()
    return criterion_labelsmoothLoss

def Get_Criterion(args):
    criterion = nn.CrossEntropyLoss().cuda()
    return criterion

def Get_Optimizer(args, model, size=1, lr=0.03):
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.0,
                          tau=args.cp,
                          gmf=0.0,
                          nesterov = False,
                          weight_decay=1e-4)

    return optimizer
'''
综合来看，这个函数的作用是根据输入的参数创建一个带有预热和余弦学习率调度的学习率调度器，并返回该调度器。
余弦学习率调度器有助于在训练过程中平滑地调整学习率，提高模型的性能。
'''
def Get_Scheduler(args, optimizer, warmup_epoch=5):

    total_steps = args.epoch * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epoch * args.iteration, total_steps)
    return scheduler

def run(rank, size):
    #dist.barrier(): 这是 PyTorch 分布式训练中的一个同步操作，用于确保所有的进程都在这个点上等待。这是为了防止不同进程在不同时间点进行操作。
    dist.barrier()
    # seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    #torch.backends.cudnn.deterministic = True: 将这个设置为 True 时，PyTorch 将会启用 CuDNN 的确定性模式。
    # 在确定性模式下，CuDNN 库会在可能的情况下尽量产生确定的结果，这对于实验的可重复性很重要。这样设置可以确保相同的模型和输入在不同运行中产生相同的结果，方便调试和验证模型的行为。
    #torch.backends.cudnn.benchmark = False: 将这个设置为 False 时，PyTorch 将禁用 CuDNN 的自动调优机制。
    # CuDNN 会根据输入数据的大小以及其他一些因素进行自动调整以提高性能，但这可能会导致运行时间的波动。通过将其设置为 False，
    # 可以确保在不同运行中获得更加一致的性能。通常，当输入数据的大小不会变化时，禁用自动调优是有益的，因为这样可以避免每次运行时重新调整。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #### Get benign train loader for each benign user, and get a backdoor dataset for the attacker
    print('get_loaders')
    #util.get_loaders(args): 调用一个名为 get_loaders 的函数，
    # 该函数返回训练和测试数据集的加载器。其中可能包括每个用户的 benign 数据集加载器、attacker 使用的 backdoor 数据集加载器等。
    train_loader_list, test_loader_list, train_attack_dataset, test_backdoor_loader, train_loader_bengin = util.get_loaders(args)

    time.sleep(5)
    #### Generate a list, recording the id of communication users of each round
    print('generate_communicate_user_list')
    #util.generate_communicate_user_list(args): 调用一个名为 generate_communicate_user_list 的函数，
    # 该函数可能用于生成每轮通信中参与通信的用户列表。这在分布式训练中很重要，因为不是所有用户都在每轮都参与通信。
    list_of_communicating_users_in_each_round = util.generate_communicate_user_list(args) #### Need to be completed

    # define neural nets model, criterion, optimizer and scheduler
    model = Get_Model(args)

    # READ_CKPT = True ### The paper Edge case use a pre-trained model!
    # if READ_CKPT :#and args.attack_type == 'edge_case':
    #     # with open("./Cifar10_VGG9_10epoch.pt", "rb") as ckpt_file:
    #     with open("./Rank0_Epoch_1499_weights.pth", "rb") as ckpt_file:
    #         ckpt_state_dict = torch.load(ckpt_file, map_location=device)
    #     model.load_state_dict(ckpt_state_dict)
    #     accuracy_of_benign_dataset = test(model, test_loader_list[0]) ### benign accuracy
    #     print('Acc of pre-trained model:',rank, accuracy_of_benign_dataset)




    fix_model = Get_Model(args)

    criterion = Get_Criterion(args)
    optimizer = Get_Optimizer(args, model, size=args.num_comm_ue, lr = args.lr)
    scheduler = Get_Scheduler(args, optimizer, warmup_epoch=args.warmup_epoch)
    LabelSmoothLoss = Get_LabelSmoothLoss(args)
    #### initialize the perturbation_for_backdoor
    perturbation_for_backdoor = np.zeros((3,32,32))
    util.save_perturbation(args, perturbation_for_backdoor)


    benign_test_acc_list = []
    backdoor_test_acc_list = []


    Reddit_word_prediction_help = None
    train_data_sets, test_data_sets, poisoned_data_for_train, test_data_poison = [], [], [], []

    for epoch in range(args.epoch):
        begin_time = time.time()

        #### load the perturbation_for_backdoor, perturbation_for_backdoor disturbance will be updated once when the attacker is scheduled.
        # perturbation_for_backdoor = util.load_perturbation(args)
        perturbation_for_backdoor = np.zeros((3,32,32))

        train(rank, model, fix_model, criterion, optimizer, scheduler, LabelSmoothLoss,
              train_loader_list, test_loader_list, epoch, device, list_of_communicating_users_in_each_round,
              attack_test_loader=test_backdoor_loader, train_attack_dataset=train_attack_dataset, v=perturbation_for_backdoor, train_loader_bengin=train_loader_bengin,
              Reddit_word_prediction_help=Reddit_word_prediction_help,
              # backdoor_optimizer=backdoor_optimizer,
              # backdoor_scheduler=backdoor_scheduler,
              train_data_sets=train_data_sets,poisoned_data_for_train=poisoned_data_for_train,
              test_data_sets=test_data_sets)
        # print(iteration,'end one Epoch')
        #### Just print the accuracy of rank 0  args.attack_target
        if rank == 0:
            #test 函数用于计算模型在良性数据集上的准确率
            accuracy_of_benign_dataset = test(model, test_loader_list[-1]) ### benign accuracy
            # print('epoch:%s,test benign acc:%s, time:%s'
            # %(epoch, round(accuracy_of_benign_dataset*100,2), time.time() - begin_time))
            #util.load_perturbation用于加载后门扰动。
            perturbation_for_backdoor = util.load_perturbation(args)
            #util.test_backdoor 用于计算模型在后门数据集上的准确率。
            accuracy_of_backdoor_dataset = util.test_backdoor(args, model, test_backdoor_loader, perturbation_for_backdoor, criterion=criterion) ### Backdoor accuracy
            #输出测试结果和运行时间，并将测试结果保存到相应的文件中。
            print('epoch:%s,test benign acc:%s,test backdoor acc:%s, sum(v):%s, time:%s'
            %(epoch, round(accuracy_of_benign_dataset*100,2), round(accuracy_of_backdoor_dataset*100,2), np.sum(perturbation_for_backdoor), time.time() - begin_time))
            #### save the acc list
            benign_test_acc_list.append(round(accuracy_of_benign_dataset*100,2))
            backdoor_test_acc_list.append(round(accuracy_of_backdoor_dataset*100,2))
            util.save_acc_file(args, rank, prefix='benign_test_acc', acc_list=benign_test_acc_list)
            util.save_acc_file(args, rank, prefix='backdoor_test_acc', acc_list=backdoor_test_acc_list)
        #### 仅在 rank 为 0 的节点上保存模型。这通常是为了保存联邦学习中的平均模型，以便用于计算后门扰动。
        if rank == 0:
            util.save_model(args.experiment_name, model, rank, epoch) #### Need to be completed

#具体而言，这个函数的操作是在输入张量 x 的右下角（从左上角起算，以距离 distance 的位置）添加一个特定的模式，
# 模式由像素值 pixel_value 组成。这个模式的形状是一个小的正方形，其右下角有四个像素点被设置为给定的 pixel_value。
def add_trigger_pattern(x, distance=2, pixel_value=255):
    shape = x.size()
    width = x.size(2)
    height = x.size(3)
    x[:,:,width - distance, height - distance] = pixel_value
    x[:,:,width - distance - 1, height - distance - 1] = pixel_value
    x[:,:,width - distance, height - distance - 2] = pixel_value
    x[:,:,width - distance - 2, height - distance] = pixel_value

    return x

def get_train_data_of_backdoor(args, user_id, train_attack_dataset, attack_test_loader, model, inputs_x, model_weights, v):
    #如果在 args.attack_type 中包含字符串 'adver'，则表示进行对抗性训练。
    if 'adver' in args.attack_type:
        model_path_list = []

        epsilon = 10.0/255.0  ###  epsilon.是扰动的阈值，用于确保生成的通用扰动的大小不超过该阈值。
        #### 调用 universal_perturbation 函数生成通用扰动，并将结果保存在变量 v 中。
        v = universal_perturbation(args, train_attack_dataset, attack_test_loader, model, model_path_list, epsilon, model_weights=model_weights,v=v)
        if np.sum(v) != 0:
            util.save_perturbation(args, v[0])

    v = util.load_perturbation(args)

    if np.sum(v) != 0 :
        v_tensor = torch.tensor(v)
        v_tensor = v_tensor.type(torch.cuda.FloatTensor)
        v_tensor = v_tensor.cuda()
        inputs_x += v_tensor

    if 'pattern' in args.attack_type:
        inputs_x = add_trigger_pattern(inputs_x, distance=2, pixel_value=1)

    return inputs_x, v

def evaluate(model, test_loader):
    model.eval()
    top1 = util.AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)
            outputs = model(data)
            acc1 = util.comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))

    return top1.avg

def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """

    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)

    return correct / total_num


'''
这个函数实现了联邦学习的一个训练循环
'''
def train(rank, model, fix_model, criterion, optimizer, scheduler, LabelSmoothLoss, train_loader_list, test_loader_list,
        epoch, device, list_of_communicating_users_in_each_round, attack_test_loader=None, train_attack_dataset=None, v=0, train_loader_bengin=None,
        Reddit_word_prediction_help=None,
        # backdoor_optimizer=None,backdoor_scheduler=None,
        train_data_sets=None,poisoned_data_for_train=None,
        test_data_sets=None):

    ######## paper "edge case" use the following # OPTIMIZER
    gamma = 0.998
    optimizer = optim_torch.SGD(model.parameters(), lr=args.lr*gamma**epoch, momentum=0.9, weight_decay=1e-4) # epoch, net, train_loader, optimizer, criterion

    model.train()
    #创建了一个模型权重的深拷贝，用于记录模型的平均权重。
    average_model_weights = copy.deepcopy(model.state_dict())
    
    # List of user ids participating at this epoch/round
    comm_user_list = list_of_communicating_users_in_each_round[epoch]
    # Get the current user and its dataloaders
    user_id = comm_user_list[rank]
    if user_id < args.attacker_user_id:
        train_loader = zip(train_loader_list[user_id], train_loader_bengin)
    else:
        train_loader = train_loader_list[user_id]
    test_loader = test_loader_list[user_id]
    #不断训练，使攻击者的后门准确率达到99%
    while True:
        for batch_idx, data in enumerate(train_loader):
            # 遍历训练数据加载器中的每个批次。
            # 数据分离：根据用户是否是攻击者，从数据中提取受毒害（poisoned）数据和良性（benign）数据。如果是攻击者用户，则从
            # data中分离poisoned_data和benign_data，并分别提取其输入poisoned_x和posioned_target，以及benign_x和benign_target。
            # 如果是良性用户，则直接提取输入数据
            # inputs_x
            # 和目标
            # targets_x。
            if user_id < args.attacker_user_id:
                poisoned_data, benign_data = data
                poisoned_x, posioned_target = poisoned_data
                poisoned_x, posioned_target = poisoned_x.to(device), posioned_target.to(device)
                benign_x, benign_target = benign_data
                benign_x, benign_target = benign_x.to(device), benign_target.to(device)
            else:
                inputs_x, targets_x = data
                inputs_x = inputs_x.to(device)
                targets_x = targets_x.to(device)

            # Add perturbation
            #后门注入：如果是攻击者用户，并且攻击类型不是 'edge_case'，则会对受毒害数据注入后门。
            # 这是通过创建模型的深拷贝 model_copy，并使用 get_train_data_of_backdoor 函数来实现的。注入后门后，
            # 从 poisoned_x 和 benign_x 中选择一半的数据用于当前训练批次。这是通过使用 torch.cat 拼接张量来实现的。
            if user_id < args.attacker_user_id and args.attack_type != 'edge_case':
                model_copy = copy.deepcopy(model)
                if np.sum(v) == 0:
                    print(f'Attack at the {epoch}-epoch --->>>>')
                    poisoned_x, v = get_train_data_of_backdoor(args, user_id, train_attack_dataset, attack_test_loader, model_copy, poisoned_x, copy.deepcopy(model.state_dict()), v)
                num_poisoned_data = args.bs//2
                inputs_x =  torch.cat((poisoned_x[0:num_poisoned_data], benign_x[num_poisoned_data:])).to(device)
                targets_x =  torch.cat((posioned_target[0:num_poisoned_data], benign_target[num_poisoned_data:])).to(device)

            output = model(inputs_x)
            loss = criterion(output, targets_x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Break and communicate if reaches the end of round/epoch
            if batch_idx == args.iteration - 1:
                break
        #训练结束判断：如果是攻击者用户，检查后门准确率是否达到 99%。如果是良性用户，直接结束训练循环。
        if user_id < args.attacker_user_id:
            accuracy_of_backdoor_dataset = util.test_backdoor(args, model, attack_test_loader, v, criterion=criterion)
            accuracy_of_benign_dataset = test(model, test_loader_list[-1])
            print('Before Scale:',round(accuracy_of_backdoor_dataset*100.0,2),round(accuracy_of_benign_dataset*100.0,2))
            model.train()

            # Attackers can finish training if the backdoor accuracy is above 99%
            if accuracy_of_backdoor_dataset*100.0 > 99.0:
                break
        else:
            # Benign users can go straight to communication after being trained for args.iteration batches. 
            break

    if user_id < args.attacker_user_id:
        # #权重缩放：如果启用了 args.weights_scale，对攻击者用户的模型进行权重缩放。这用于增加攻击者模型在联邦平均中的权重，以便更好地模拟攻击。
        if args.weights_scale:
            attack_w = copy.deepcopy(model.state_dict())
            attack_w_scale = util.weights_scale(attack_w, average_model_weights, float(args.num_comm_ue))
            model.load_state_dict(attack_w_scale)

        # Test accuracies
        accuracy_of_backdoor_dataset = util.test_backdoor(args, model, attack_test_loader, v, criterion=criterion)
        accuracy_of_benign_dataset = test(model, test_loader_list[-1]) ### benign accuracy
        print('After Scale:',round(accuracy_of_backdoor_dataset*100.0,2),round(accuracy_of_benign_dataset*100.0,2))
        
        # if norm_bound is used, attacker will projected its model's norm samll than s_norm

        if args.norm_bound:
            NDC(args, model, average_model_weights, user_id, s_norm=args.s_norm_attack)

        model.train()

    SyncAllreduce(model, rank, args.num_comm_ue)
    return

def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):

    target_lr = args.lr * args.bs * scale * args.num_comm_ue / 128

    lr = None
    if args.warmup_epoch and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def init_processes(rank, size, fn, ip_address, master_port):
    ######################## Get IP address.
    print(args.master_node)
    if args.run_slurm:
        os.environ['MASTER_ADDR'] = args.master_node 
    else:
        os.environ['MASTER_ADDR'] = args.ip_address
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group('gloo', rank=rank, world_size=size)

    torch.cuda.manual_seed(1)
    fn(rank, size)

if __name__ == "__main__":
    rank = args.rank
    world_size = args.num_comm_ue
    master_port = args.master_port

    print(rank)
    print(args)
    ######### Assign Ranks to different GPUs
    GRU_list = [i for i in args.GPU_list]
    increase_tmp = (world_size+1)//len(GRU_list)
    ranks_list = np.arange(0, world_size).tolist()

    rank_group = []
    for rank_id in range(len(GRU_list)):
        if rank_id == len(GRU_list)-1:
            ranks = ranks_list[rank_id*increase_tmp:]
        else:
            ranks = ranks_list[rank_id*increase_tmp:(rank_id+1)*increase_tmp]
        rank_group.append(ranks)

    if args.run_slurm:
        pass
    else:
        for group_id in range(len(GRU_list)):
            if args.rank in set(rank_group[group_id]):
                os.environ["CUDA_VISIBLE_DEVICES"] = GRU_list[group_id]
                break

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    init_processes(rank, world_size, run, args.ip_address, master_port)
