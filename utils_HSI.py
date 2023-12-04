# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
import matplotlib.pyplot as plt
from scipy import io
import imageio
import os
import re
import torch
import numpy as np
import h5py
import shutil
from AdversarialNetwork import RandomLayer, AdversarialNetwork, calc_coeff
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Data
import math

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device

def seed_worker(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        if 'HyRANK' in dataset:
            return io.loadmat(dataset)  # HyRANK
        elif 'Houston' in dataset:
            return h5py.File(dataset)   # Houston
        elif 'Pavia' in dataset:
            return io.loadmat(dataset)  # Pavia
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #target = target[ignored_mask] -1
    # target = target[ignored_mask]
    # prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes)) # cm[i, j]代表的是真实标签为 i 但模型预测为 j 的样本数量

    results["Confusion_matrix"] = cm

    # 下面四个指标都是(class_num, )
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate(recall)
    TPR = TP/(TP+FN)
    results["TPR"] = TPR
    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
    # 对于每个类别 i，精确率为 cm[i, i] / np.sum(cm[:, i])，即在所有预测为第 i 类别的样本中，真正
    # 属于第 i 类别的比例；召回率为 cm[i, i] / np.sum(cm[i, :])，即在所有真实属于第 i 类别的样本中，
    # 成功识别为第 i 类别的比例；
    # F1 = 2 * precision * recall / (precision + recall)
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1
    results["F1_scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    results["prediction"] = prediction
    results["label"] = target

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1_scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion_matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion_matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1_scores"]
        kappa = results["Kappa"]

    #label_values = label_values[1:]
    vis.heatmap(cm, opts={'title': "Confusion_matrix", 
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion_matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1_scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt) # (行坐标，列坐标), 注意这里是非0的坐标，也就是不是背景的类别才有训练的必要
    X = list(zip(*indices)) # the list of (x,y)
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random':
        if train_size == 1: # target data
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)] # return [[x1, x2, ..., xn], [y1, y2, ..., yn]]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label)) # [[x0,y0,c0], [x1,y1,c1]...]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0],train_indices[1],train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0],test_indices[1],test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))

    return train_gt, test_gt, train_set, test_set


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def bulid_adversarial_network(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    if args.randomLayer:
        random_layer = RandomLayer([model_output_num, class_num], 1024)
        ad_net = AdversarialNetwork(1024, 512)
        random_layer.cuda()
        
    else:
        random_layer = None
        if args.methodOfDAN=='DANN':
            ad_net = AdversarialNetwork(model_output_num, 128)
        else:
            ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return random_layer, ad_net

def Entropy(input_):
    return torch.sum(-input_ * torch.log(input_ + 1e-5), dim=1)

def grl_hook(coeff):
    def fun1(grad):
        return - coeff * grad.clone()
    return fun1


def DANN(features, ad_net):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if dc_target.is_cuda:
        dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    feature = input_list[0]
    softmax_output = input_list[1].detach()
    
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        print(f"random_out's shape is {random_out.shape}")
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
        print(f"ad_out's shape is {ad_out.shape}")

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if feature.is_cuda:
        dc_target = dc_target.cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask

        target_mask = torch.ones_like(entropy)
        target_mask[:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def makeFolder(args):
    os.makedirs(args.output_path)
    os.mkdir(args.output_path + '/code')
    prefixPath = args.output_path + '/code'
    if not os.path.exists(prefixPath):
        os.mkdir(prefixPath)

    rootdir = './'
    list = os.listdir(rootdir)
    for line in list:
        filepath = os.path.join(rootdir, line)
        if os.path.isfile(filepath):
            if filepath.endswith('.py'):
                try:
                    shutil.copyfile(filepath, os.path.join(prefixPath, filepath.split('/')[-1]))
                except:
                    continue
        elif line == 'model':
            shutil.copytree(filepath, os.path.join(prefixPath, 'model'))



def CELoss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def ConsistencyLoss(logits_s, logits_w, class_acc, p_target=None, p_model=None, name='ce',
                     T=1.0, p_cutoff=0.99, use_hard_labels=True, use_DA=False):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1) # batch * 7
        if use_DA:
            if p_model == None:
                p_model = torch.mean(pseudo_label.detach(), dim=0)
            else:
                p_model = p_model * 0.999 + torch.mean(pseudo_label.detach(), dim=0) * 0.001
            pseudo_label = pseudo_label * p_target / p_model
            pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

        max_probs, max_idx = torch.max(pseudo_label, dim=-1) # batch * 1, batch * 1
        print(f"max_probs = {max_probs}")
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        # mask = max_probs.ge(p_cutoff).float()  # fixed threshold value
        # mask = max_probs.ge(p_cutoff * class_acc[max_idx]).float()  # convex
        # mask = max_probs.ge(p_cutoff * torch.pow((class_acc[max_idx] + 1.) / 2., 2)).float()  # convex2, rise rate is slower than convex
        mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))).float()  # convex
        # mask = max_probs.ge((1 - p_cutoff)  + p_cutoff * torch.pow((class_acc[max_idx] + 1.) / 2., 2)).float()  # residual
        mask_ = mask.data.clone().detach()
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff).long()

        if use_hard_labels:
            masked_loss = CELoss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = CELoss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), mask_, select, max_idx.long(), p_model

class GradReverse(Function):
    # def __init__(self, lambd):
    #     super().__init__()
    #     self.lambd = lambd

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(torch.tensor(lambd))
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0]
        return (grad_output * -lambd), None
        # return (grad_output.neg(),)

def grad_reverse(x, lambd=1.0):
    # return GradReverse(lambd)(x)
    return GradReverse.apply(x, lambd)

def getGraphdata(source_share, bs, target_share, target=True):
    DEVICE = torch.device('cuda:0')
    segments = torch.reshape(torch.tensor(range(bs)),(-1, int(math.sqrt(bs)))) # segments的shape是(sqrt(10), sqrt(10))
    src_edge = torch.tensor(getEdge(source_share, segments)).t().contiguous() # src_edge.shape (2, 595), 记录的是这个batch里边每个样本的边的关系
    source_share_graph = Data(x=source_share,edge_index=src_edge).to(DEVICE)
    if target == True:
        tar_edge = torch.tensor(getEdge(target_share, segments)).t().contiguous()
        target_share_graph = Data(x=target_share,edge_index=tar_edge).to(DEVICE)
    else:
        target_share_graph =  0
    return source_share_graph, target_share_graph

def getGraphdataOneDomain(share, bs):
    DEVICE = torch.device('cuda:0')
    segments = torch.reshape(torch.tensor(range(bs)),(-1, int(math.sqrt(bs))))
    edge = torch.tensor(getEdge(share, segments)).t().contiguous()
    share_graph = Data(x=share, edge_index=edge).to(DEVICE)
    return share_graph

def getGraphdata_ClassGuided(share, src_labels, tar_labels, mask):
    DEVICE = torch.device('cuda:0')
    edge = torch.tensor(getEdge_ClassGuided(src_labels, tar_labels, mask)).t().contiguous()
    if edge.size(0) == 0:
        return None
    share_graph = Data(x=share, edge_index=edge).to(DEVICE)
    return share_graph

def getEdge(image, segments, compactness=300, sigma=3.):
    coo = set()
    dire = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    for i in range(1, segments.shape[0]):
        for j in range(1, segments.shape[1]):
            for dx, dy in dire:
                if -1 < i + dx < segments.shape[0] and \
                        -1 < j + dy < segments.shape[1] and \
                        segments[i, j] != segments[i + dx, j + dy]:
                    coo.add((segments[i, j], segments[i + dx, j + dy]))
    coo = np.asarray(list(coo))
    return coo

def getEdge_ClassGuided(src_labels, tar_labels, mask):
    coo = set()
    src_labels = src_labels.type(torch.int)
    src_node, tar_node = src_labels.size(0), tar_labels.size(0)
    src_dict, tar_dict = {}, {}
    for i in range(src_node):
        if src_labels[i].item() not in src_dict and mask[i] > 0:
            src_dict[src_labels[i].item()] = []
        if mask[i] > 0:
            src_dict[src_labels[i].item()].append(i)
    for i in range(tar_node):
        if tar_labels[i].item() not in tar_dict and mask[i] > 0:
            tar_dict[tar_labels[i].item()] = []
        if mask[i] > 0:
            tar_dict[tar_labels[i].item()].append(src_node + i)

    for src_class, src_indexs in src_dict.items():
        if src_class in tar_dict:
            tar_indexs = tar_dict[src_class]
            [coo.add((x, y)) for x in src_indexs for y in tar_indexs]
            [coo.add((y, x)) for x in src_indexs for y in tar_indexs]

    coo = np.asarray(list(coo))
    return coo

def zero_grads(*args):
    for optimizer in args:
        optimizer.zero_grad()
    
def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))