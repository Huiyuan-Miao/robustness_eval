# test the model's ability to recognize objects with conflicting texture and shape. 
import re
import os
import sys
import random
import time
import numpy
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils_2 import *
from alexnet import *
from shapeTextureBiasHelper import helper, probabilities_to_decision
import pandas as pd

def main(whichModel,num_classes,dataDir):

    #### Parameters ####################################################################################################
    model_path = '/home/tonglab/Miao/pycharm/AlexNetSurroundSuppression/'

    train_batch_size = 128
    val_batch_size = 128
    start_epoch = 0
    num_epochs = 50
    save_every_epoch = 10
    initial_learning_rate = 1e-2
    gpu_ids = [0,1]

    #### Create/Load model #############################################################################################
    # 1. If pre-trained models used without pre-trained weights. e.g., model = models.vgg19()
    # 2. If pre-trained models used with pre-trained weights. e.g., model = models.vgg19(pretrained=True)
    # 3. If our models used.
    ####################################################################################################################

    # model = models.alexnet(pretrained=False)
    # model = AlexNet_divNorm4(num_classes=num_classes,c = 0.0001)
    model = AlexNet_v1_divNormV2(num_classes=num_classes)


    if len(gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0]).cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)

    #### Resume from checkpoint
    try:
        os.mkdir(model_path)
    except:
        pass

    load_path = os.path.join(model_path, whichModel) +'checkpoint.pth.tar'
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch'] + 1

        if len(gpu_ids) <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict
        else: # 2. single-GPU or -CPU to Multi-GPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'module.' in k:
                    name = k
                else:
                    name = 'module.' + k
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['initial_lr'] = initial_learning_rate
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        print("train from start")

        # with torch.no_grad():
        #     model.features[0].weight.copy_(checkpoint['state_dict']['features.0.weight'])
        #     model.features[0].bias.copy_(checkpoint['state_dict']['features.0.bias'])
        # model.features[0].weight.requires_grad = False # this is for Alexnet_v3 freeze layer 1 weight as the original alexnet weight
        # model.features[0].bias.requires_grad = False # this is for Alexnet_v3

    #### Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[61], gamma=0.1, last_epoch=start_epoch-1)
    lr_scheduler.step()

    #### Data loader ###################################################################################################
    # 0, n01622779, Owl; 1, n02123045, Cat; 2, n02129165, Lion; 3, n02132136, Bear; 4, n02326432, Hare;
    # 5, n02342885, Hamster; 6, n02410509, Bison; 7, n02504458, Elephant; 8, n02690373, Airliner;
    # 9, n03594945, Jeep; 10, n04147183, Schooner; 11, n04273569, Speedboat; 12, n04285008, Sports car;
    # 13, n04344873, Couch; 14, n04380533, Table lamp, 15, n04398044, Teapot
    ####################################################################################################################

    train_dataset = torchvision.datasets.ImageFolder(
        # "/home/tonglab/Documents/Data/ILSVRC2012/images/train_16",
        '/home/tonglab/Datasets/shapeTextureBias/style-transfer-preprocessed-512',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = torchvision.datasets.ImageFolder(
    val_dataset = ImageFolderWithPaths(
        # "/home/tonglab/Documents/Data/ILSVRC2012/images/val_16",
        '/home/tonglab/Datasets/shapeTextureBias/style-transfer-preprocessed-512',
        transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1280, shuffle=False, num_workers=4, pin_memory=True)


    #### Train/Val #####################################################################################################
    for ssnr in [1]:
        mapping = {'airplane': 0, 'bear': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                   'car': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'dog': 10, 'elephant': 11,
                   'keyboard': 12, 'knife': 13, 'oven': 14, 'truck': 15}
        stat_file = open(os.path.join(model_path, whichModel, 'evalGaussianNoiseImg.txt'), 'a+')

        target, decision_from_16_classes, confidence_of_16_classes = val(val_loader, model, loss_function, optimizer, stat_file, gpu_ids)
        target = np.array(target.cpu())
        confidence_target = np.zeros(np.shape(target))
        confidence_confusion = np.zeros(np.shape(target))
        conf = np.zeros((decision_from_16_classes.size))
        for i in range(decision_from_16_classes.size):
            dir = val_loader.dataset.imgs[i][0]
            conf[i] = mapping[checkConfCate(dir)]
            confidence_target[i] = confidence_of_16_classes[i, target[i]]
            confidence_confusion[i] = confidence_of_16_classes[i, int(conf[i])]
        dic = {'shape': target, 'texture': conf, 'answer': decision_from_16_classes,
               'confidence_shape': confidence_target, 'confidence_texture': confidence_confusion}
        scipy.io.savemat(os.path.join(model_path, whichModel, 'testingShapeBias_stats.mat'), dic)
        stat_file = open(os.path.join(model_path, whichModel, 'testingShapeBias_stats.txt'), 'a+')
        for i in range(len(decision_from_16_classes)):
            stat_str = 'target ({:.0f}) confusion ({:.0f}) answer ({:.0f})'. \
                format(target[i], conf[i], decision_from_16_classes[i])
            stat_file.write(stat_str + '\n')
        stat_file.close()
        stat_file = open(os.path.join(model_path, whichModel, 'testingShapeBias_stats2.txt'), 'a+')
        acc = sum(target == decision_from_16_classes) / len(target)
        target_sub = target[target != decision_from_16_classes]
        ans_sub = decision_from_16_classes[target != decision_from_16_classes]
        conf_sub = conf[target != decision_from_16_classes]
        conf_percent = sum(conf_sub == ans_sub) / len(target)
        stat_str = 'accuracy ({:.6f}) confusion ({:.6f})'. \
            format(acc, conf_percent)
        stat_file.write(stat_str + '\n')
        stat_file.close()

        df = pd.read_csv(os.path.join(model_path, whichModel, 'testingShapeBias_stats.txt'), header=None, sep=' ')
        df_sub = df[[1, 3, 5]].rename(columns={1: 'target', 3: 'confusion', 5: 'answer'})
        for col in df_sub.columns:
            df_sub[col] = df_sub[col].apply(lambda x: cleanNum(x)).astype(float)
        df_sub = np.array(df_sub)
        n1 = np.sum(np.logical_and(df_sub[:, 2] == df_sub[:, 1], df_sub[:, 2] == df_sub[:, 0]))
        n2 = np.sum(np.logical_and(df_sub[:, 2] != df_sub[:, 1], df_sub[:, 2] == df_sub[:, 0]))
        n3 = np.sum(np.logical_and(df_sub[:, 2] == df_sub[:, 1], df_sub[:, 2] != df_sub[:, 0]))
        n4 = np.sum(np.logical_and(df_sub[:, 2] != df_sub[:, 1], df_sub[:, 2] != df_sub[:, 0]))
        shapebias = n2 / (n2 + n3)
        stat_file = open(os.path.join(model_path, whichModel, 'testingShapeBias_stats3.txt'), 'a+')
        stat_str = 'ans = shape & ans = texture ({:.0f}) ans = shape & ans ~= texture ({:.0f}) ' \
                   'ans ~= shape & ans = texture ({:.0f}) ans ~= shape & ans ~= texture ({:.0f}) ' \
                   'shape bias ({:.6f})'. \
            format(n1, n2, n3, n4, shapebias)
        stat_file.write(stat_str + '\n')
        stat_file.close()

        df_sub = df_sub[df_sub[:, 1] != df_sub[:, 0], :]
        n2 = np.sum(np.logical_and(df_sub[:, 2] != df_sub[:, 1], df_sub[:, 2] == df_sub[:, 0])) / len(df_sub)
        n3 = np.sum(np.logical_and(df_sub[:, 2] == df_sub[:, 1], df_sub[:, 2] != df_sub[:, 0])) / len(df_sub)
        n4 = np.sum(np.logical_and(df_sub[:, 2] != df_sub[:, 1], df_sub[:, 2] != df_sub[:, 0])) / len(df_sub)
        stat_file = open(os.path.join(model_path, whichModel, 'testingShapeBias_stats3.txt'), 'a+')
        stat_str = 'shape ~= texture ({:.0f}) ans = shape & ans ~= texture ({:.6f}) ' \
                   'ans ~= shape & ans = texture ({:.6f}) ans ~= shape & ans ~= texture ({:.6f})'. \
            format(len(df_sub), n2, n3, n4)
        stat_file.write(stat_str + '\n')
        stat_file.close()




def val(val_loader, model, loss_function, optimizer, stat_file,gpu_ids):
    mapping_dict = {'airplane': 0, 'bear': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
                    'car': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'dog': 10, 'elephant': 11,
                    'keyboard': 12, 'knife': 13, 'oven': 14, 'truck': 15}
    model.eval()

    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0.
    correct1, correct5 = [0] * len(val_loader.dataset.samples), [0] * len(val_loader.dataset.samples) # Correct

    for batch_index, (input, target, paths, indices) in enumerate(val_loader):
        if len(gpu_ids) >= 1:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = model(input)
        softmax_output_numpy = np.array(nn.Softmax(dim=1)(output).detach().cpu())
        mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
        decision_from_16_classes = np.zeros((output.size(0)))
        confidence_of_16_classes = np.zeros((output.size(0), 16))
        # conf = np.zeros((output.size(0),1))
        for i in range(output.size(0)):
            decision_from_16_classes[i] = mapping_dict[mapping.probabilities_to_decision(softmax_output_numpy[i])]
            confidence_of_16_classes[i, :] = mapping.probabilities_of_decision(softmax_output_numpy[i])
    return target, decision_from_16_classes, confidence_of_16_classes


def cleanNum(a):
    temp = re.findall('[0-9.]*',a)
    num = temp[1]
    return num

def checkConfCate(dir):
    t = dir.split('/')[-1]
    t1 = t.split('-')
    t2 = t1[-1].split('.')
    t3 = re.findall('[a-z.]*',t1[0])[0]
    t4 = re.findall('[a-z.]*',t2[0])[0]
    return t4

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True) # top5 glitch

        return correct, num_correct

def add_gauss_noise(images,ssnr):
    noised_images = torch.zeros_like(images)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(images.size(0)):
        image = images[i,:,:,:]
        sigma = (1-ssnr)/2/3 # I don't understand but hojin defined in such way...
        signal = (image-0.5)*ssnr + 0.5
        noise = np.tile(np.random.normal(0,sigma,(1,images.size(2),images.size(3))),(images.size(1),1,1,))
        noise = torch.from_numpy(noise).float().to(images.device)
        noised_image = signal + noise
        noised_image[noised_image > 1] = 1
        noised_image[noised_image < 0] = 0
        noised_image = normalize(noised_image)
        noised_images[i] = noised_image
    return noised_images


if __name__ == '__main__':
    # for i in ['ckpt_latest_trainForClassification.pth','ckpt_latest_finetuneWithDualErr.pth']:
    for i in [1]:
        if i == 1:
            whichModel = 'AlexNet_/'
            num_classes = 1000
            dataDir = '/home/tonglab/Datasets/imagenet1000/'
            main(whichModel = whichModel,num_classes = num_classes,dataDir = dataDir)
        elif i == 2:
            whichModel = 'AlexNet_blurToClear_place365/'
            num_classes = 365
            dataDir = '/home/tonglab/Datasets/places365_standard/'
            main(whichModel = whichModel,num_classes = num_classes,dataDir = dataDir)
