from shutil import copyfile
import datetime
import math
import torch

from torch.autograd import Variable
import logging
import numpy as np
import copy

import random
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import os

from copy import deepcopy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)

class Helper:
    def __init__(self, params):
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.benign_test_data = None
        self.poisoned_data = None
        self.poisoned_test_data = None

        self.params = params
        self.best_loss = math.inf
    #这个函数看起来是用于比较两组神经网络权重之间的差异
    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                ## 计算weight1和weight2中相应层之间的差异
                difference[name] = layer.data - weight2[name].data
                ## 将差异张量扁平化并附加到res列表中
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat

    @staticmethod
    def get_l2_norm(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)
        ## 计算差异张量的L2范数
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        ## 将差异张量转换为NumPy数组，然后计算其L2范数
        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())

        return l2_norm, l2_norm_np

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):
        # 计算缩放因子 scale，该因子用于确保L2范数不超过给定的阈值 norm_bound。
        # 如果 l2_norm 小于等于 norm_bound，则 scale 为 1.0，否则为 l2_norm / norm_bound
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        scale =  max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)

        return weight_difference, l2_norm

    def grad_mask(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        hidden = model.init_hidden(helper.params['batch_size'])
        for participant_id in range(len(dataset_clearn)):
            train_data = dataset_clearn[participant_id]
            if helper.params['task'] == 'word_predict':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['sequence_length'])
                ntokens = 50000
                #内层循环遍历数据集中的每个批次
                for batch in data_iterator:
                    model.train()
                    data, targets = helper.get_batch(train_data, batch)
                    #将模型的隐藏状态重新封装，这样可以防止梯度在时间维度上的累积，有助于减轻梯度消失或爆炸的问题。
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    #这里使用的是词语预测任务的交叉熵损失。
                    class_loss = criterion(output.view(-1, ntokens), targets)
                    class_loss.backward(retain_graph=True)
            elif helper.params['task'] == 'sentiment':
                for inputs, labels in train_data:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    hidden = helper.repackage_hidden(hidden)
                    inputs = inputs.type(torch.LongTensor).cuda()
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward(retain_graph=True)
            else:
                raise ValueError("Unkonwn task")
        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
            grad_list = torch.cat(grad_list).cuda()
            # # 选取梯度最小的一部分索引
            #grad_list 是一个包含模型参数梯度绝对值的一维张量。
            # -1 * grad_list 将梯度绝对值取负，这是因为torch.topk返回的是最大值的索引，而我们实际上想要的是最小值的索引。
            # torch.topk 函数接收两个参数：要查找最大值的输入张量和要选择的最大值的数量。
            # int(len(grad_list) * ratio) 计算要选择的最大值的数量，其中 ratio 是在总梯度数中选择的比例。
            # torch.topk(-1 * grad_list, int(len(grad_list) * ratio)) 返回一个元组，第一个元素是所选最大值的负值，第二个元素是相应的索引。
            # 我们使用 _ 来接收第一个元素，而 indices 接收第二个元素。
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            ## 将索引转换为CPU上的NumPy数组
            indices = list(indices.cpu().numpy())
            count = 0
            #为模型参数设置掩码 只允许更新梯度较小的参数位置上的掩码为1，其余位置为0表示不更新梯度
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    count_list = list(range(count, count + len(parms.grad.abs().view(-1))))
                    #index_list = list(set(count_list).intersection(set(indices))) 计算 count_list 和 indices 之间的交集，得到当前参数的梯度索引。
                    index_list = list(set(count_list).intersection(set(indices)))
                    mask_flat = np.zeros( count + len(parms.grad.abs().view(-1))  )

                    mask_flat[index_list] = 1.0
                    mask_flat = mask_flat[count:count + len(parms.grad.abs().view(-1))]
                    mask = list(mask_flat.reshape(parms.grad.abs().size()))

                    mask = torch.from_numpy(np.array(mask, dtype='float32')).cuda()
                    mask_grad_list.append(mask)
                    count += len(parms.grad.abs().view(-1))
        else:
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
        model.zero_grad()
        return mask_grad_list

    def grad_mask_cv(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        ## 遍历每个参与者的数据集，执行前向传播，计算损失，并执行反向传播以计算梯度。retain_graph=True确保保留计算图以进行多次反向传播。
        for participant_id in range(len(dataset_clearn)):

            train_data = dataset_clearn[participant_id]

            for inputs, labels in train_data:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
                    ## 计算每个层参数梯度的绝对值之和
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                    k_layer += 1

            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))
                    #通过切片操作，从全层梯度掩码中提取当前参数的梯度掩码。
                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)


                    k_layer += 1

        model.zero_grad()
        return mask_grad_list


    def grad_mask_gpt2(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        for i in range(len(dataset_clearn)):
            train_dataloader = dataset_clearn[i]
            for batch_id, batch in enumerate(train_dataloader):
                model.train()

                data1, data2 = batch['input_ids'], batch['attention_mask']

                data1 = [x.unsqueeze(0) for x in data1]
                data2 = [x.unsqueeze(0) for x in data2]

                data1 = torch.cat(data1).transpose(0,1)
                data2 = torch.cat(data2).transpose(0,1)

                input_ids = data1[:,0:0+helper.params['sequence_length']]
                att_masks = data2[:,0:0+helper.params['sequence_length']]

                target = data1[:,1:1+helper.params['sequence_length']].reshape(-1)

                input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()

                output = model(input_ids, attention_mask=att_masks).logits

                loss = criterion(output.contiguous().view(-1, self.n_tokens), target)
                loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        mask_grad_list = []

        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

        model.zero_grad()
        return mask_grad_list

    def lr_decay(self, epoch):

        return 1


    def average_shrink_models(self, weight_accumulator, target_model, epoch, wandb):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        #调用self.lr_decay方法，根据当前训练轮数epoch计算学习率lr。
        lr = self.lr_decay(epoch)

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            #计算每一层的更新参数
            update_per_layer = weight_accumulator[name] * \
                               (1/self.params['partipant_sample_size']) * \
                               lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)

            data.add_(update_per_layer.cuda())

        return True
