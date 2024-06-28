import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
import math
from torch_geometric.data import Data
from train import DEVICE
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'model'))
from backbone import ResNet50Fc as ResNet50
from utils_HSI import *
import torch.nn.init as init

class vgg16(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, init_weights=True, batch_norm=True):
        super(vgg16, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [32, 32, 64, 64],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'yao': [64, 128, 256, 256]
        }
        layers = []
        for v in cfg['D']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, stride=1, kernel_size=3)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.features(x)
            t, c, w, h = x.size()
        return t * c * w * h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # raw x = (bs, c, ps, ps)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class PredictorCNN(nn.Module):
    def __init__(self, in_dim=1024, num_class=7, prob=0.5, init_type='kaiming_normal'):
        super(PredictorCNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 4096)
        self.bn1_fc = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_class)
        self.bn_fc3 = nn.BatchNorm1d(num_class)
        self.prob = prob
        self.init_type = init_type
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x

class PredictorGCN(nn.Module):
    def __init__(self, in_dim,num_class,dropout=0.5, init_type='kaiming_normal'):
        super(PredictorGCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1024, bias=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1024, num_class, bias=True)
        self.init_type = init_type
        self._initialize_weights()

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def _initialize_weights(self):
        if self.init_type == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif self.init_type == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
        elif self.init_type == 'xavier_uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels):
        super(Topology_Extraction, self).__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        # x_temp_1 = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x_temp_2 = x

        return x

class GraphMCD(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, **kwargs):
        super(GraphMCD, self).__init__()
        self.classes = num_classes
        #@ CNN (backbone / generator / feature extractor)
        self.backbone = vgg16(in_channels, num_classes, patch_size)
        self.backbone_output_dim = self.backbone._get_final_flattened_size()
        #@ GCN
        self.src_gcn = Topology_Extraction(self.backbone_output_dim)
        self.tar_gcn = Topology_Extraction(self.backbone_output_dim)
        self.gcn_output_dim = self.src_gcn.conv2.out_channels
        #@ classifiers
        self.classifierCNN1 = PredictorCNN(in_dim=self.backbone_output_dim, num_class=self.classes, init_type='kaiming_normal')
        # self.classifierCNN2 = PredictorCNN(in_dim=self.backbone_output_dim, num_class=self.classes, init_type='xavier_uniform')
        self.classifierGCN1 = PredictorGCN(in_dim=self.gcn_output_dim, num_class=self.classes, init_type='kaiming_uniform')
        # self.classifierGCN2 = PredictorGCN(in_dim=self.gcn_output_dim, num_class=self.classes, init_type='xavier_normal')

    def forward(self, source, target=None, useOne=False, reverse=False):
        out = self.backbone(source)
        bs = out.shape[0]
        cnn_pred1 = self.classifierCNN1(out, reverse)
        if self.training:
            share_graph = getGraphdataOneDomain(out, bs)
            share_graph = self.src_gcn(share_graph)
            gcn_pred1 = self.classifierGCN1(share_graph, reverse)
            if useOne:
                # return cnn_pred1, gcn_pred1
                return cnn_pred1, gcn_pred1, out, share_graph
            else:
                tar_out = self.backbone(target)
                tar_cnn_pred1 = self.classifierCNN1(tar_out, reverse)
                tar_share_graph = getGraphdataOneDomain(tar_out, bs)
                tar_share_graph = self.tar_gcn(tar_share_graph)
                tar_gcn_pred1 = self.classifierGCN1(tar_share_graph, reverse)
                # return cnn_pred1, gcn_pred1, tar_cnn_pred1, tar_gcn_pred1
                return cnn_pred1, gcn_pred1, tar_cnn_pred1, tar_gcn_pred1,\
                out, share_graph, tar_out, tar_share_graph
        else:
            return cnn_pred1, None

class CrossGraph(nn.Module):
    def __init__(self, in_channels):
        super(CrossGraph, self).__init__()
        self.gcn = Topology_Extraction(in_channels)

    def forward(self, src_feat, tar_feat, src_labels, tar_labels, mask):
        feat = torch.cat((src_feat, tar_feat), dim=0)
        share_graph = getGraphdata_ClassGuided(feat, src_labels, tar_labels, mask)
        if share_graph == None:
            return feat
        share_graph = self.gcn(share_graph)
        return share_graph
