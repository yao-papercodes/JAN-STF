from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model import GMCD
import numpy as np
from utils_HSI import *
from utils_PL import *
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
from collections import Counter
import os
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import scipy.io as io
from sklearn.metrics import classification_report
import torch.nn as nn
import datetime
import shutil
import pickle


parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='../Datasets/S-H/',
                    help='the path to load the data')
parser.add_argument('--log_path', type=str, default='./logs',
                    help='the path to load the data')
parser.add_argument('--output_path', type=str, default='./exp',
                    help='the path to save this exp data.')

parser.add_argument('--source_name', type=str, default='Dioni',
                    help='the name of the source dir, can automaticly change by programe')
parser.add_argument('--target_name', type=str, default='Loukia',
                    help='the name of the test dir, can automaticly change by programe')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=12,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lr_ad', type=float, default=1e-5,
                    help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=100,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=1233, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2 weight decay')
parser.add_argument('--num_epoch', type=int, default=300,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=10, help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.05, help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=10,
                    help='multiple of data augmentation. It can be calculated automatically during the training process, and is not predetermined!')
# parser.add_argument('--checkpoint', type=str, default='/media/sx636/WD/YAO/HSICC/TGRS_23/SEJGA/exp/2023-06-26-16:55:12/params_Houston13_Acc7775_Kappa6459.pkl', help='checkpoint path')
parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint path')
parser.add_argument('--isTest', type=bool, default=False, help='whether is just concede the inference stage')

#! ToAlign
parser.add_argument('--useToAlign', type=bool, default=False, help='whether to use ToAlign.')

#! HDA head
parser.add_argument('--useHDA', type=bool, default=False, help='whether to use HDA head to classify.')
parser.add_argument('--lambda_hda', type=float, default=1.0, help="hyperparameter of HDA head classification loss")

#! adverserial network
parser.add_argument('--useDAN', type=bool, default=False, help='whether to use DAN Loss')
parser.add_argument('--lambda_dan', type=float, default=0.1, help="hyperparameter of adverserial loss")

#! maximum classifier discrepancy
parser.add_argument('--useMCD', type=bool, default=True, help='whether to use MCD Loss')
parser.add_argument('--numK', type=int, default=15, help='update time of Generator')
parser.add_argument('--lambda_reverse', type=int, default=1, help='hyperparameter to conctrol the RGL, can be automatictly change by programe.')

#! class alignment
parser.add_argument('--useCAL', type=bool, default=True, help='whether to use class alignment loss')
parser.add_argument('--thresholdCAL', type=bool, default=True, help='the threshold of pseudo labels genertated in CAL')
parser.add_argument('--lambda_cal', type=int, default=1e-5, help='hyperparameter to conctrol the CAL.')

#! class alignment
parser.add_argument('--useCGDAN', type=bool, default=True, help='whether to use cross graph domain adverserial network')
parser.add_argument('--thresholdCGDAN', type=bool, default=True, help='the threshold of pseudo labels genertated in CGDAN')
parser.add_argument('--lambda_cgdan', type=int, default=1.0, help='hyperparameter to conctrol the CGDAN.')
parser.add_argument('--methodOfDAN', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
parser.add_argument('--useFeatureType', type=str, default='GCN', choices=['CNN', 'GCN'])
parser.add_argument('--randomLayer', type=bool, default=False, help='whether to use random')

#! pseudo labels
parser.add_argument('--usePLF', type=bool, default=False, help='real parameter to judge whether to use pseudo label generation')
parser.add_argument('--startPLEpoch', type=int, default=0, help='start epoch of using pseudo label generation')
parser.add_argument('--usePL', type=bool, default=False, help='whether to use pseudo labels (change in computing)')
parser.add_argument('--plAlg', type=str, default='fixmatch', choices=['fixmatch', 'flexmatch', 'freematch', 'softmatch'])
parser.add_argument('--lambda_pl', type=float, default=1.0, help="hyperparameter of pseudo label loss")
parser.add_argument('--classNum', type=int, default=7, help='class number, no need to predefine')

#! target consistency
parser.add_argument('--useTarCons', type=bool, default=False, help="whether to use the target consistency module to align the cnn & gcn prediction")
parser.add_argument('--lambda_tar_cons', type=float, default=0.1, help="hyperparameter of target consistency")

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--rotate_augmentation', action='store_true', default=True,
                    help="Random rotate (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=True,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

args = parser.parse_args()
DEVICE = get_device(args.cuda)

def train(epoch, model_GMCD, model_CG, model_Discriminator, random_layer, num_epoch):

    #@ Build the optimizers
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    LEARNING_RATE_DAN = args.lr_ad / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    optimizer_GMCD = optim.SGD(model_GMCD.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)
    if args.useCGDAN:
        optimizer_CG = optim.SGD(model_CG.parameters(), lr=LEARNING_RATE, momentum=args.momentum,weight_decay = args.l2_decay)
        optimizer_Dis = optim.SGD(model_Discriminator.parameters(), lr=LEARNING_RATE_DAN, momentum=args.momentum,weight_decay = args.l2_decay)
        optimizer_Dis, lr_dis = lr_scheduler_withoutDecay(optimizer_Dis, lr=LEARNING_RATE_DAN)
    if (epoch-1)%10==0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))

    #@ Initialize the metrics
    global writer
    Loss_Cls_CNN, Loss_Cls_GCN, Loss_Dis, Loss_PL = 0, 0, 0, 0
    Correct_Src_CNN, Correct_Src_GCN, Correct_Tar_CNN, Correct_Tar_GCN = 0, 0, 0, 0
    Correct_GCN, Mask_GCN, Counter_GCN, Loss_CAL = 0, 0, Counter(), 0
    # Correct_CGDAN is for adverserial accuracy, Correct_CNN is for pseudo label accuracy from CNN
    Correct_CGDAN, Correct_CNN, Mask_CNN, Counter_CNN, Loss_CGDAN = 0, 0, 0, Counter(), 0
    len_tar_temp = 0

    iter_source = data_prefetcher(train_loader) # 定义在 if __name__ == '__main__'这个函数里边的变量，就是全局变量，是可以被其他函数访问的
    iter_target = data_prefetcher(train_tar_loader) # prefetcher的作用就是先缓存数据
    if args.usePL:
        iter_target_pl = iter(train_tar_loader_pl)
  
    num_iter = len_src_loader
    bs = train_loader.batch_size

    model_GMCD.train()
    if args.useCGDAN:
        model_CG.train()
        model_Discriminator.train()
    for i in range(1, num_iter):

        if 0 < (len_tar_train_dataset-i*bs) < bs or i % len_tar_train_loader == 0:
            iter_target = data_prefetcher(train_tar_loader)
            if args.usePL:
                iter_target_pl = iter(train_tar_loader_pl)
        index_src, data_src, label_src = iter_source.next()
        index_tar, data_tar, label_tar = iter_target.next()
        label_src = label_src - 1
        label_tar = label_tar - 1

        # zero_grads(optimizer_G, optimizer_C1, optimizer_C2, optimizer_SGCN, optimizer_TGCN)
        optimizer_GMCD.zero_grad()
        if args.useCGDAN:
            optimizer_CG.zero_grad()
            optimizer_Dis.zero_grad()

        out = model_GMCD(data_src, useOne=True)
        src_cnn_pred, src_gcn_pred = out[0], out[1]
        src_cnn_feat, src_gcn_feat = out[2], out[3]
        loss_cls_cnn = F.nll_loss(F.log_softmax(src_cnn_pred, dim=1), label_src.long())
        loss_cls_gcn = F.nll_loss(F.log_softmax(src_gcn_pred, dim=1), label_src.long())
        loss_cls = loss_cls_cnn + loss_cls_gcn
        if args.useMCD:
            model_GMCD.classifierCNN1.set_lambda(args.lambda_reverse)
            model_GMCD.classifierGCN1.set_lambda(args.lambda_reverse)
            out = model_GMCD(data_tar, useOne=True, reverse=True)
            tar_cnn_pred, tar_gcn_pred = out[0], out[1]
            tar_cnn_feat, tar_gcn_feat = out[2], out[3]
            loss_dis = -discrepancy(tar_cnn_pred, tar_gcn_pred)
            # loss_dis.backward()
            # optimizer_GMCD.step()
        else:
            # optimizer_GMCD.step()
            loss_dis = torch.tensor(0)
            out = model_GMCD(data_tar, useOne=True)
            tar_cnn_pred, tar_gcn_pred = out[0], out[1]
            tar_cnn_feat, tar_gcn_feat = out[2], out[3]

        #! -------- add the class alignment generation module ---------
        if args.useCAL:
            tar_gcn_pred_ = tar_gcn_pred.clone().detach()
            if args.plAlg == 'softmatch':
                tar_gcn_pred_ = torch.softmax(tar_gcn_pred_, dim=-1)
                tar_gcn_pred_ = plgs[2].dist_align(probs_x_ulb=tar_gcn_pred_)
            mask = plgs[0].masking(logits_x_ulb=tar_gcn_pred_, idx_ulb=index_tar, p_cutoff=args.thresholdCAL)
            pseudo_label = gen_ulb_targets(logits=tar_gcn_pred_)
            loss_cal = class_alignment_loss2(src_cnn_feat, tar_cnn_feat[mask>0], label_src, pseudo_label[mask>0])
            Loss_CAL += loss_cal.item()
            pred_gcn_mask = pseudo_label[mask>0]
            pred_gcn_label = label_tar[mask>0]
            Correct_GCN += pred_gcn_mask.eq(pred_gcn_label).cpu().sum()
            Mask_GCN += mask.sum()
            Counter_GCN.update(pred_gcn_mask.tolist())
        else:
            loss_cal = torch.tensor(0.)
            Loss_CAL += loss_cal.item()
        #! ---------------------------------------------

        #! -------- add the cross graph domain adverserial network module ---------
        if args.useCGDAN:
            #@ 1. Cross Graph Convolution
            tar_cnn_pred_ = tar_cnn_pred.clone().detach()
            if args.plAlg == 'softmatch':
                tar_cnn_pred_ = torch.softmax(tar_cnn_pred_, dim=-1)
                tar_cnn_pred_ = plgs[3].dist_align(probs_x_ulb=tar_cnn_pred_)
            mask2 = plgs[1].masking(logits_x_ulb=tar_cnn_pred_, idx_ulb=index_tar, p_cutoff=args.thresholdCGDAN)
            pseudo_label = gen_ulb_targets(logits=tar_cnn_pred_)
            gcn_feat_cg = model_CG(src_cnn_feat, tar_cnn_feat, label_src, pseudo_label, mask2)
            if gcn_feat_cg.size(1) == model_GMCD.backbone_output_dim:
                gcn_feat_cg = torch.cat((src_gcn_feat, tar_gcn_feat))

            # sub_size = int(mask2.sum())
            pred_cnn_mask = pseudo_label[mask2>0]
            pred_cnn_label = label_tar[mask2>0]
            Correct_CNN += pred_cnn_mask.eq(pred_cnn_label).cpu().sum()
            Mask_CNN += mask2.sum()
            Counter_CNN.update(pred_cnn_mask.tolist())
            # src_gcn_feat = src_gcn_feat_cg
            # tar_gcn_feat[mask2>0] = tar_gcn_feat_cg

            #@ 2. Domain Adaption
            # feature = torch.cat((src_gcn_feat_cg, tar_gcn_feat_cg), dim=0)
            feature = gcn_feat_cg
            pred = torch.cat((src_cnn_pred, tar_cnn_pred), dim=0)
            softmax = nn.Softmax(dim=1)
            softmax_output = softmax(pred)
            if args.methodOfDAN == 'CDAN-E':
                entropy = Entropy(softmax_output)
                loss_cgdan = CDAN([feature, softmax_output], model_Discriminator, entropy, calc_coeff(num_iter * (epoch - 1) + i), random_layer)
            elif args.methodOfDAN == 'CDAN':
                loss_cgdan = CDAN([feature, softmax_output], model_Discriminator, None, None, random_layer)
            elif args.methodOfDAN == 'DANN':
                loss_cgdan = DANN(feature, model_Discriminator)
            Loss_CGDAN += loss_cgdan.item()

            #@ 3. Compute the accuracy
            if args.methodOfDAN == 'CDAN' or args.methodOfDAN == 'CDAN-E':
                softmax_output = nn.Softmax(dim=1)(pred)
                if args.randomLayer:
                    random_out = random_layer.forward([feature, softmax_output])
                    adnet_output = model_Discriminator(random_out.view(-1, random_out.size(1)))
                else:
                    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1)) # softmax_output's shape is (batchSize, 7, 1) feature's shape is (batchSize, 1, 384)
                    adnet_output = model_Discriminator(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
            elif args.methodOfDAN == 'DANN':
                adnet_output = model_Discriminator(feature)

            adnet_output = adnet_output.cpu().data.numpy()
            adnet_output[adnet_output > 0.5] = 1
            adnet_output[adnet_output <= 0.5] = 0
            Correct_CGDAN += np.sum(adnet_output[:args.batch_size]) + (args.batch_size - np.sum(adnet_output[args.batch_size:]))
            # Correct_CGDAN += np.sum(adnet_output[:sub_size]) + (sub_size - np.sum(adnet_output[sub_size:]))
        else:
            Correct_CGDAN = 0.
            loss_cgdan = torch.tensor(0.)
            Loss_CGDAN = loss_cgdan.item()
        #! ------------------------------------------------------------------------
        loss_ = loss_cls + loss_dis + args.lambda_cal * loss_cal + args.lambda_cgdan * loss_cgdan
        loss_.backward()
        optimizer_GMCD.step()
        if args.useCGDAN:
            optimizer_CG.step()
            optimizer_Dis.step()

        Loss_Cls_CNN += loss_cls_cnn.item()
        Loss_Cls_GCN += loss_cls_gcn.item()
        Loss_Dis += loss_dis.item()
        pred_cnn = src_cnn_pred.data.max(1)[1]  # 如果是取[0]的话就是取坐标, [1]是取值
        Correct_Src_CNN += pred_cnn.eq(label_src.data.view_as(pred_cnn)).cpu().sum()
        pred_gcn = src_gcn_pred.data.max(1)[1]
        Correct_Src_GCN += pred_gcn.eq(label_src.data.view_as(pred_gcn)).cpu().sum()
        len_tar_temp += tar_cnn_pred.shape[0]

        if len_tar_train_dataset - len_tar_temp >= 0:
            pred_cnn_tar = tar_cnn_pred.data.max(1)[1]
            Correct_Tar_CNN += pred_cnn_tar.eq(label_tar.data.view_as(pred_cnn_tar)).cpu().sum()
            pred_gcn_tar = tar_gcn_pred.data.max(1)[1]
            Correct_Tar_GCN += pred_gcn_tar.eq(label_tar.data.view_as(pred_gcn_tar)).cpu().sum()
            len_tar = len_tar_temp

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format( epoch+1, i * len(data_src), len_src_dataset, 100. * i / len_src_loader))
            print('Loss_Cls_C1: {:.3f},  Loss_Cls_C2: {:.3f}, Loss_Dis: {:.3f} Loss_CAL: {:.3f} Loss_CGDAN: {:.3f}'.format(loss_cls_cnn.item(), loss_cls_gcn.item(), loss_dis.item(), loss_cal.item(), loss_cgdan.item()))
            if args.useCAL:
                print(f"{mask.sum()} target samples have been assigned pseudo labels to in CAL")
            if args.useCGDAN:
                print(f"{mask2.sum()} target samples have been assigned pseudo labels to in CGDAN ")

    Acc_Src_CNN = Correct_Src_CNN.item() / len_src_dataset
    Acc_Src_GCN = Correct_Src_GCN.item() / len_src_dataset
    Acc_Tar_CNN = Correct_Tar_CNN.item() / len_tar
    Acc_Tar_GCN = Correct_Tar_GCN.item() / len_tar
    Acc_CGDAN = Correct_CGDAN.item() / (2 * args.batch_size * num_iter) if Correct_CGDAN > 0 else 0

    print('[epoch: {:4}]  Train Accuracy: {:.2%}, {:.2%}, {:.2%}, {:.2%} | train sample number: {:6}'.format(epoch+1, Acc_Src_CNN, Acc_Src_GCN, Acc_Tar_CNN, Acc_Tar_GCN, len_src_dataset))
    writer.add_scalars('Loss_group', {'Cls_CNN': Loss_Cls_CNN/len_src_loader, 'Cls_GCN': Loss_Cls_GCN/len_src_loader, 'Discrepence': Loss_Dis/len_src_loader, 'CAL': Loss_CAL/len_src_loader, 'CGDAN': Loss_CGDAN/(2*args.batch_size*num_iter)}, epoch)
    if args.useCAL or args.useCGDAN:
        writer.add_scalars('PL_Confi_group', {'GCN': Correct_GCN/Mask_GCN if args.useCAL else 0., 'CNN': Correct_CNN/Mask_CNN if args.useCGDAN else 0.}, epoch)
        if epoch % args.log_interval == 0:
            Counter_list, Name_list = [], []
            if args.useCAL:
                Counter_list.append(Counter_GCN)
                Name_list.append("GCN")
            if args.useCGDAN:
                Counter_list.append(Counter_CNN)
                Name_list.append("CNN")
            fig = Visualize_Collection_Pseudo_Labels_Category(Counter_list, Name_list, class_num=args.classNum, mode='quantity')
            writer.add_figure('PL_Quantity_group', fig, epoch)
    return model_GMCD, model_CG, model_Discriminator, random_layer, Acc_Src_CNN, Acc_Src_GCN, Acc_Tar_CNN, Acc_Tar_GCN, Acc_CGDAN

def test(model):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for _, data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            label = label - 1
            out = model(data)
            pred = out[0].data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(out[0], dim = 1), label.long()).item() # sum up batch loss

        loss /= len_tar_loader
        print('Testing...')
        print('{} set: Average test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, | Test sample number: {:6}'.format(
            args.target_name, loss, correct, len_tar_dataset,
            100. * correct / len_tar_dataset, len_tar_dataset))
    return correct.item() / len_tar_dataset, correct, pred_list, label_list


if __name__ == '__main__':

    seed_worker(args.seed)
    dt = datetime.datetime.now()
    folder = dt.strftime('%Y-%m-%d-%H:%M:%S')
    args.output_path = os.path.join(args.output_path, folder)
    makeFolder(args)
    args.log_path = args.output_path
    args.save_path = args.output_path

    acc_test_list = [0. for i in range(args.num_trials)]
    acc_class_test_list = [{} for i in range(args.num_trials)]
    kappa_test_list = [0. for i in range(args.num_trials)]
    for flag in range(args.num_trials):
        #@ some hyperparameters depend on the dataset name
        if 'Houston' in args.data_path:
            args.source_name='Houston13'
            args.target_name='Houston18'
            args.lr = 1e-3
            # args.lr = 1e-2
            args.lambda_reverse = 1
            args.lambda_cal = 4e-6
            args.lambda_cgdan = 1e-1
            args.thresholdCAL = 0.99
            args.thresholdCGDAN = 0.99
        elif 'HyRANK' in args.data_path:
            args.source_name='Dioni'
            args.target_name='Loukia'
            args.lr = 1e-1
            args.lambda_reverse = 5
            args.lambda_cal = 1e-3
            args.lambda_cgdan = 1e-3
            args.thresholdCAL = 0.90
            args.thresholdCGDAN = 0.90
        elif 'Pavia' in args.data_path:
            args.source_name='paviaU'
            args.target_name='paviaC'
            args.lr = 1e-2
            args.lambda_reverse = 1
            args.lambda_cal = 1e-3
            args.lambda_cgdan = 1e-2
            args.thresholdCAL = 0.95
            args.thresholdCGDAN = 0.90
        elif 'S-H' in args.data_path:
            args.source_name='Hangzhou'
            args.target_name='Shanghai'
            args.lr = 1e-2
            args.lambda_reverse = 1
            args.lambda_cal = 1e-3
            args.lambda_cgdan = 1e-2
            args.thresholdCAL = 0.99
            args.thresholdCGDAN = 0.95


        img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                args.data_path)
        img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                args.data_path)

        sample_num_src = len(np.nonzero(gt_src)[0]) # 统计的非零像素点的个数
        sample_num_tar = len(np.nonzero(gt_tar)[0])
        # args.re_ratio = min(int(sample_num_tar/(sample_num_src * args.training_sample_ratio)), int(1/args.training_sample_ratio))
        args.re_ratio = 1
        tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
        training_sample_tar_ratio = tmp if tmp < 1 else 1
        print(f"SD Sample Rate = {args.training_sample_ratio:.2%}")
        print(f"TD Sample Rate = {training_sample_tar_ratio:.2%}")
        print(f"re_ratio = {args.re_ratio}")
        print(f"learning rate = {args.lr}")

        num_classes=gt_src.max()
        N_BANDS = img_src.shape[-1]
        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                            'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        r = int(hyperparams['patch_size']/2) + 1
        img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric') # raw: (W,H,C) => (W+2r, H+2r, C)
        img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric') # 镜像填充, 所以不用使用特殊数值
        gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
        gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))

        train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
        train_gt_tar, _, _, _ = sample_gt(gt_tar, training_sample_tar_ratio, mode='random')
        img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar
        if tmp < 1:
            for i in range(args.re_ratio-1):
                img_src_con = np.concatenate((img_src_con,img_src))
                train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
                # img_tar_con = np.concatenate((img_tar_con,img_tar))
                # train_gt_tar_con = np.concatenate((train_gt_tar_con,train_gt_tar))
        args.classNum = int(gt_src.max())
        
        # Generate the dataset
        hyperparams_train = hyperparams.copy()

        train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
        g = torch.Generator() # 随机数生成类，训练和测试期间生成可重复的伪随机数字序列
        g.manual_seed(args.seed)
        train_loader = data.DataLoader(train_dataset,
                                        batch_size=hyperparams['batch_size'],
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True)
        train_tar_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
        train_tar_loader = data.DataLoader(train_tar_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=True)
        #@ use pseudo label generation
        args.usePL = False
        train_tar_dataset_pl = None
        train_tar_loader_pl = None

        test_dataset = HyperX(img_tar, test_gt_tar, flag='Test', **hyperparams)
        test_loader = data.DataLoader(test_dataset,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'])                      
        len_src_loader = len(train_loader) # batch_size为批次
        len_tar_train_loader = len(train_tar_loader)
        len_src_dataset = len(train_loader.dataset) # 总数量
        len_tar_train_dataset = len(train_tar_loader.dataset)
        len_tar_dataset = len(test_loader.dataset)
        len_tar_loader = len(test_loader)

        print(hyperparams)
        '''
        train samples和train tar samples样本数量差不多的原因:
        因为利用HyperX在构建source和target的dataset的时候, 主要是利用输入的train_gt里边的非零标签构建的,
        而train_gt又是经过一定比例采样的, source的数据会比target少很多, source在经过re_ratio之后,数量就
        跟target差不多了。
        '''
        print("train samples :", len_src_dataset)
        print("train tar samples :", len_tar_train_dataset)

        correct, acc = 0, 0
        model_GMCD = GMCD.GraphMCD(img_src.shape[-1],num_classes=int(gt_src.max()), patch_size=hyperparams['patch_size']).to(DEVICE)
        if args.useCGDAN:
            model_CG = GMCD.CrossGraph(model_GMCD.backbone_output_dim).to(DEVICE)
            random_layer, model_Discriminator = bulid_adversarial_network(args, model_GMCD.backbone_dim if hyperparams['useFeatureType'] == 'CNN' else 32, int(gt_src.max()))
        else:
            model_CG, random_layer, model_Discriminator = None, None, None
        if args.checkpoint != 'None':
            dic = torch.load(args.checkpoint)
            model_param_dict = dic['model_GMCD']
            model_dict = model_GMCD.state_dict()
            model_param_dict = {k: v for k, v in model_param_dict.items() if k in model_dict}
            model_dict.update(model_param_dict)
            model_GMCD.load_state_dict(model_dict)

        log_dir = os.path.join(args.log_path, args.source_name+'_lr_'+str(args.lr)+'_' + str(int(flag+1))+'times')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            shutil.copyfile('./tb.sh', os.path.join(log_dir, 'tb.sh'))
        writer = SummaryWriter(log_dir)

        for epoch in range(args.num_epoch):
            if epoch == args.startPLEpoch and args.usePLF:
                args.usePL = True
                hyperparams['usePL'] = True
                train_tar_dataset_pl = HyperX(img_tar_con, train_gt_tar_con, useST=True, **hyperparams)
                train_tar_loader_pl = data.DataLoader(train_tar_dataset_pl,
                                        pin_memory=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        batch_size=hyperparams['batch_size'],
                                        shuffle=True)
                plgs = construct_plg(args.plAlg, len_tar_train_dataset, args.classNum)
            
            if args.useCAL or args.useCGDAN:
                plgs = construct_plg(args.plAlg, len_tar_train_dataset, args.classNum)

            if not args.isTest:
                model_GMCD, model_CG, model_Discriminator, random_layer, Acc_Src_CNN, Acc_Src_GCN, Acc_Tar_CNN, Acc_Tar_GCN, Acc_CGDAN = \
                    train(epoch, model_GMCD, model_CG, model_Discriminator, random_layer, args.num_epoch)
            if epoch % args.log_interval == 0:
                Test_Acc, t_correct, pred, label = test(model_GMCD)
                if t_correct > correct:
                    correct = t_correct
                    acc = Test_Acc
                    if acc > 0.5:
                        acc_test_list[flag] = acc
                        results = {}
                        metrics_hand_cal = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=int(gt_src.max()))
                        class_acc_str = classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar, digits=4)
                        class_acc_dict = classification_report(np.concatenate(pred),np.concatenate(label),target_names=LABEL_VALUES_tar, digits=4, output_dict=True)
                        results['class_acc_str'] = class_acc_str
                        results['class_acc_dict'] = class_acc_dict
                        results['metrics_hand_cal'] = metrics_hand_cal
                        acc_class_test_list[flag] = results
                        model_save_path = os.path.join(args.save_path, 'params_'+args.source_name+'_Acc'+str(int(acc*10000))+'_Kappa'+str(int(metrics_hand_cal['Kappa']*10000))+'.pkl')
                        kappa_test_list[flag] = metrics_hand_cal['Kappa']
                        print(class_acc_str)
                        print(f"kappa: {kappa_test_list[flag]:.2%}")
                        torch.save({'model_GMCD': model_GMCD.state_dict(), 'model_CG': model_CG.state_dict() if args.useCGDAN else None, 'model_Discriminator': model_Discriminator.state_dict() if args.useCGDAN else None}, model_save_path)

            print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
                args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset ))

            writer.add_scalars('Accuracy_group', {'CNN_Train': Acc_Src_CNN, 'GCN_Train': Acc_Src_GCN, 'CNN_Train_Tar': Acc_Tar_CNN, 'GCN_Train_Tar': Acc_Tar_GCN, 'CNN_Test': Test_Acc, 'CGDAN': Acc_CGDAN}, epoch)
        with open(os.path.join(args.save_path,'train_times_'+args.source_name+'.pickle'), 'wb') as f:
            pickle.dump({'acc_test_list': acc_test_list, 'kappa_test_list': kappa_test_list, 'acc_class_test_list': acc_class_test_list, 'lr':args.lr, 'class_num': args.classNum}, f)
        # io.savemat(os.path.join(args.save_path,'train_times_'+args.source_name+'.mat'), {'acc_test_list': acc_test_list, 'kappa_test_list': kappa_test_list, 'acc_class_test_list': acc_class_test_list, 'lr':args.lr, 'class_num': args.classNum})