# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import copy
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
import random
from PIL import Image
from scipy.linalg import sqrtm
from scipy.ndimage import rotate
from RandAugment import RandAugment
import torchvision.transforms as transforms
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file

DATASETS_CONFIG = {
        'Houston13': {
            'img': 'Houston13.mat',
            'gt': 'Houston13_7gt.mat',
            },
        'Houston18': {
            'img': 'Houston18.mat',
            'gt': 'Houston18_7gt.mat',
            },
        'Shanghai': {
            'img': 'Shanghai.mat',
            'gt': 'Shanghai_gt.mat',
            },
        'Hangzhou': {
            'img': 'Hangzhou.mat',
            'gt': 'Hangzhou_gt.mat',
            },
        'Dioni': {
            'img': 'Dioni.mat',
            'gt': 'Dioni_gt_out68.mat',
            },
        'Loukia': {
            'img': 'Loukia.mat',
            'gt': 'Loukia_gt_out68.mat',
            },
        'paviaU': {
            'img': 'paviaU.mat',
            'gt': 'paviaU_7gt.mat',
            },
        'paviaC': {
            'img': 'paviaC.mat',
            'gt': 'paviaC_7gt.mat',
            },
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder# + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        # Load the image
        img = open_file(folder + 'Houston13.mat')['ori_data'][:]
        img = np.transpose(img, (1, 2, 0))

        rgb_bands = (43, 21, 11) #! 这个是什么用法，rgb光的通道下标吗？

        gt = open_file(folder + 'Houston13_7gt.mat')['map'][:]

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7"]

        ignored_labels = [0]
    elif dataset_name == 'Houston18':
        # Load the image
        img = open_file(folder + 'Houston18.mat')['ori_data'][:]
        img = np.transpose(img, (1, 2, 0))

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Houston18_7gt.mat')['map'][:]

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7"]

        ignored_labels = [0]
    elif dataset_name == 'Shanghai':
        # Load the image
        img = open_file(folder + 'Shanghai.mat')['DataCube1']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Shanghai_gt.mat')['gt1']

        label_values = ["1", "2", "3"]

        ignored_labels = [0]
    elif dataset_name == 'Hangzhou':
        # Load the image
        img = open_file(folder + 'Hangzhou.mat')['DataCube2']
        # ['DataCube1']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Hangzhou_gt.mat')['gt2']
        # ['gt1']

        label_values = ["1", "2", "3"]

        ignored_labels = [0]
    elif dataset_name == 'Dioni':
        # Load the image
        img = open_file(folder + 'Dioni.mat')['ori_data']

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Dioni_gt_out68.mat')['map']

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7",
                        "8", "9", "10",
                        "11", "12"]

        ignored_labels = [0]
    elif dataset_name == 'Loukia':
        # Load the image
        img = open_file(folder + 'Loukia.mat')['ori_data'][:]

        rgb_bands = (43, 21, 11)

        gt = open_file(folder + 'Loukia_gt_out68.mat')['map'][:]

        label_values = ["1", "2", "3",
                        "4", "5",
                        "6", "7",
                        "8", "9", "10",
                        "11", "12"]
                        
        ignored_labels = [0] 
    elif dataset_name == 'paviaU':
        # Load the image
        img = open_file(folder + 'paviaU.mat')['ori_data'][:]

        rgb_bands = [20,30,30]

        gt = open_file(folder + 'paviaU_7gt.mat')['map'][:]

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]     
    elif dataset_name == 'paviaC':
        # Load the image
        img = open_file(folder + 'paviaC.mat')['ori_data'][:]

        rgb_bands = [20,30,30]

        gt = open_file(folder + 'paviaC_7gt.mat')['map'][:]

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']
                    
        ignored_labels = [0]

    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img= img.reshape((m*n,-1))
    img = img/img.max()
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))
    img_temp = np.expand_dims(img_temp,axis=1)
    img_temp = img_temp.repeat(d,axis=1)
    img_temp[img_temp==0]=1
    img = img/img_temp
    img = np.reshape(img,(m,n,-1))
    
    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, useST=False, flag='Train', **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data # (W, H, C)
        self.label = gt  # (W, H)
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.rotate_augmentation = hyperparams['rotate_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        self.usePL = hyperparams['usePL']
        self.useST = useST
        self.flag = flag

        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        '''
        self.indices: np.array, shape=(len, 2), 存储的是gt里边非零值的坐标
        self.labels: list, length = len, 存储的是gt里边非零值坐标对应的类别值
        '''
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p]) # 保证在非padding，原图区域中
        self.labels = [self.label[x,y] for x,y in self.indices]
        
        state = np.random.get_state()
        np.random.shuffle(self.indices)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def rotate(*arrays):
        angle = random.randint(1, 3)
        arrays = [np.rot90(arr, k=angle) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise
    
    def random_flip(self, data): # strong transform 1
        """
        随机翻转数据
        :param data: numpy数组，形状为(12, 12, 100)
        :return: 随机翻转后的数据
        """
        flip_code = np.random.randint(0, 2, size=3) * 2 - 1
        return np.flip(data, axis=2)
    
    def random_rotate(self, data, angle_range=30): # strong transform 2
        """
        随机旋转数据
        :param data: numpy数组，形状为(12, 12, 100)
        :param angle_range: 旋转角度范围
        :return: 随机旋转后的数据
        """
        angle = np.random.uniform(-angle_range, angle_range)
        return rotate(data, angle, reshape=False)

    def random_brightness(self, image, max_delta=0.5): # strong transform 3
        delta = np.random.uniform(-max_delta, max_delta)
        return np.clip(image + delta, 0.0, 1.0)

    def random_contrast(self, image, lower=0.5, upper=1.5): # strong transform 4
        alpha = np.random.uniform(lower, upper)
        gray = image.mean(axis=2, keepdims=True)
        return np.clip(alpha * (image - gray) + gray, 0.0, 1.0)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]
        data_s = np.copy(data) # 用于强数据增强使用，shape是(W, H, C)
        label_s = np.copy(label)

        if self.flag == 'Train':
            if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
                data, label = self.flip(data, label)
            if self.rotate_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
                data, label = self.rotate(data, label)
            if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)
        
        # data_s = np.copy(data)  # 用于强数据增强使用，shape是(W, H, C)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views) numpy is (W, H, C), torch is (C, W, H)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32') # 因为这个取出的patch是一个正方形，所以W和H不用care
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)   # patch_size * patch_size
        label = torch.from_numpy(label) # patch_size * patch_size
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
            
        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        if self.usePL and self.useST:
            data_s, label_s = self.flip(data_s, label_s)
            data_s, label_s = self.rotate(data_s, label_s)
            data_s = self.radiation_noise(data_s)
            # data_s = self.mixture_noise(data_s, label_s)
            data_s = np.asarray(np.copy(data_s).transpose((2, 0, 1)), dtype='float32')
            data_s = torch.from_numpy(data_s)
            return i, data, label, data_s
        return i, data, label # data(patchsize * patchsize) label(a value)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.index, self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.index = self.index.cuda(non_blocking=True)
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        index = self.index
        data = self.data
        label = self.label

        self.preload()
        return index.long(), data, label