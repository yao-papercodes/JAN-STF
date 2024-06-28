# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch.nn import functional as F
import numpy as np
from collections import Counter
from copy import deepcopy

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time

class FixedThresholdingHook():
    """
    Common Fixed Threshold used in fixmatch, uda, pseudo label, et. al.
    """

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, p_cutoff=0.99, *args, **kwargs):
        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
        return mask

class FlexMatchThresholdingHook():
    """
    Adaptive Thresholding in FlexMatch
    """
    def __init__(self, ulb_dest_len, num_classes, thresh_warmup=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ulb_dest_len = ulb_dest_len
        self.num_classes = num_classes
        self.thresh_warmup = thresh_warmup
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs):
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                # if -1 in wo_negative_one.keys():
                #     wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

    @torch.no_grad()
    def masking(self, logits_x_ulb, idx_ulb, softmax_x_ulb=True, p_cutoff=0.99, *args, **kwargs):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
            # probs_x_ulb = self.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(p_cutoff)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()

        return mask

class FreeMatchThresholing():
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) # / self.num_classes  局部类别阈值
        self.label_hist = torch.ones((self.num_classes)) # / self.num_classes   直方图统计类别数目
        self.time_p = self.p_model.mean()   # 总阈值
    
    @torch.no_grad()
    def update(self, probs_x_ulb, use_quantile=False, clip_thresh=False):
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()

        if clip_thresh: # 截断
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

    

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask


class SoftMatchWeightingHook():
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.9, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
            self.prob_max_var_t =  torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, probs_x_ulb):
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # per_class默认为False
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2)))) # mask (bs, 1)
        return mask

class DistAlignEMAHook():
    """
    Distribution Alignment Hook for conducting distribution alignment
    """
    def __init__(self, num_classes, momentum=0.999, p_target_type='uniform', p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target_type, p_target) # p_target是用lb来坐的，p_model才是用ulb来坐的
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    @torch.no_grad()
    def dist_align(self, probs_x_ulb, probs_x_lb=None):
        # update queue
        self.update_p(probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6) # p_model's shape=(c, ), p_target's shape=(c, )
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    

    @torch.no_grad()
    def update_p(self, probs_x_ulb, probs_x_lb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        if self.p_model == None:
            self.p_model = torch.mean(probs_x_ulb, dim=0)
        else:
            self.p_model = self.p_model * self.m + torch.mean(probs_x_ulb, dim=0) * (1 - self.m)

        if self.update_p_target:
            assert probs_x_lb is not None
            self.p_target = self.p_target * self.m + torch.mean(probs_x_lb, dim=0) * (1 - self.m)
    
    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        update_p_target = False
        if p_target_type == 'uniform':
            p_target = torch.ones((self.num_classes, )) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.ones((self.num_classes, ))/ self.num_classes
            update_p_target = True
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
        
        return update_p_target, p_target


def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist

def compute_prob(self, logits):
    return torch.softmax(logits, dim=-1)

def gen_ulb_targets(logits, use_hard_label=True, T=1.0, softmax=True, label_smoothing=0.0): # softmax: whether to compute softmax for logits, input must be logits
    """
    generate pseudo-labels from logits/probs

    Args:
        algorithm: base algorithm
        logits: logits (or probs, need to set softmax to False)
        use_hard_label: flag of using hard labels instead of soft labels
        T: temperature parameters
        softmax: flag of using softmax on logits
        label_smoothing: label_smoothing parameter
    """

    logits = logits.detach()
    if use_hard_label:
        # return hard label directly
        pseudo_label = torch.argmax(logits, dim=-1)
        if label_smoothing:
            pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
        return pseudo_label
    
    # return soft label
    if softmax:
        # pseudo_label = torch.softmax(logits / T, dim=-1)
        pseudo_label = compute_prob(logits / T)
    else:
        # inputs logits converted to probabilities already
        pseudo_label = logits
    return pseudo_label
        
def consistency_loss(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

def construct_plg(alg_name, len_tar_train_dataset, num_classes):
    print(f"===> use \033[31m{alg_name}\033[0m to generate pseudo labels")
    if alg_name == 'fixmatch':
        plg = FixedThresholdingHook()
        plg2 = FixedThresholdingHook()
        return plg, plg2
    elif alg_name == 'flexmatch':
        plg = FlexMatchThresholdingHook(ulb_dest_len=len_tar_train_dataset, num_classes=num_classes, thresh_warmup=True)
        plg2 = FlexMatchThresholdingHook(ulb_dest_len=len_tar_train_dataset, num_classes=num_classes, thresh_warmup=True)
        return plg, plg2
    elif alg_name == 'freematch':
        plg = FreeMatchThresholing(num_classes=num_classes)
        plg2 = FreeMatchThresholing(num_classes=num_classes)
        return plg, plg2
    elif alg_name == 'softmatch':
        plg = SoftMatchWeightingHook(num_classes=num_classes) # pseudo labels generation
        plg2 = SoftMatchWeightingHook(num_classes=num_classes)
        da = DistAlignEMAHook(num_classes=num_classes, p_target_type='uniform')
        da2 = DistAlignEMAHook(num_classes=num_classes, p_target_type='uniform')
        return plg, plg2, da, da2

def construct_one_plg(alg_name, len_tar_train_dataset, num_classes):
    print(f"===> use \033[31m{alg_name}\033[0m to generate pseudo labels")
    if alg_name == 'fixmatch':
        plg = FixedThresholdingHook()
        return plg, None
    elif alg_name == 'flexmatch':
        plg = FlexMatchThresholdingHook(ulb_dest_len=len_tar_train_dataset, num_classes=num_classes, thresh_warmup=True)
        return plg, None
    elif alg_name == 'freematch':
        plg = FreeMatchThresholing(num_classes=num_classes)
        return plg, None
    elif alg_name == 'softmatch':
        plg = SoftMatchWeightingHook(num_classes=num_classes) # pseudo labels generation
        da = DistAlignEMAHook(num_classes=num_classes, p_target_type='uniform')
        return plg, da

def construct_plg_test(len_tar_train_dataset, num_classes):
    plgs_test = [[FixedThresholdingHook(), FlexMatchThresholdingHook(ulb_dest_len=len_tar_train_dataset, num_classes=num_classes, thresh_warmup=True),\
                  FreeMatchThresholing(num_classes=num_classes), SoftMatchWeightingHook(num_classes=num_classes)] for i in range(4)]
    return plgs_test


def Visualize_Collection_Pseudo_Labels_Category(pseudo_counters, counter_names, class_num, mode='proportion'):
    assert mode in ['proportion', 'quantity']
    props = []
    for classifier_id in range(len(pseudo_counters)):
        counter = pseudo_counters[classifier_id]
        if mode == 'proportion':
            total = sum(counter.values())
            prop = [counter[c] / total for c in range(class_num)]
        elif mode == 'quantity':
            prop = [counter[c] for c in range(class_num)]
        props.append(prop)

    data = [[f'C{i}'] for i in range(class_num)]
    for class_id in range(class_num):
        for classifier_id in range(len(pseudo_counters)):
            data[class_id].append(props[classifier_id][class_id])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    df=pd.DataFrame(data,columns=["Landmark"] + counter_names)
    ax = df.plot(x="Landmark", y=counter_names, kind="bar", figsize=(9,8), ax=ax, title='collection of category pl', grid=True, rot=360)
    
    if mode == 'proportion':
        return fig, props

    return fig


class Class_Alignment():
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = momentum
        self.src_proto = None
        self.tar_proto = None
    
    def update(self, source, target, src_labels, tar_labels):
        # Computing prototypes for each class as the mean of the extracted features
        source_dict = {}
        target_dict = {}
        source_dict = dict(zip(source, src_labels))
        final_source_dict = {}
        processed_keys = set()
        for key in source_dict:
            if key in processed_keys:
                continue
            counter = 1
            sum_ = key
            for inner_key in source_dict:
                if not torch.all(torch.eq(key, inner_key)) and source_dict[key].item() == source_dict[inner_key].item():
                    processed_keys.add(inner_key)
                    counter += 1
                    sum_ = sum_ + inner_key
            prototype = sum_ / counter
            final_source_dict[source_dict[key].item()] = prototype

        target_dict = dict(zip(target, tar_labels))
        final_target_dict = {}
        processed_keys = set()
        for key in target_dict:
            if key in processed_keys:
                continue
            counter = 1
            sum_ = key
            for inner_key in target_dict:
                if not torch.all(torch.eq(key, inner_key)) and target_dict[key].item() == target_dict[inner_key].item():
                    processed_keys.add(inner_key)
                    counter += 1
                    sum_ = sum_ + inner_key
            prototype = sum_ / counter
            final_target_dict[target_dict[key].item()] = prototype

        # Adding squared euclidean distances of prototypes of same classes. 
        # If a class is present in the source domain but not in the target domain
        # it is ignored
        sum_dists = 0

        for key in final_source_dict:
            if key in final_target_dict:
                s = ((final_source_dict[key] - final_target_dict[key]) ** 2).sum(axis=0)
                sum_dists = sum_dists + s

        return sum_dists

def class_alignment_loss(source, target, src_labels, tar_labels):
    # Computing prototypes for each class as the mean of the extracted features
    source_dict = {}
    target_dict = {}
    source_dict = dict(zip(source, src_labels))
    final_source_dict = {}
    processed_keys = set()
    for key in source_dict:
        if key in processed_keys:
            continue
        counter = 1
        sum_ = key
        for inner_key in source_dict:
            if not torch.all(torch.eq(key, inner_key)) and source_dict[key].item() == source_dict[inner_key].item():
                processed_keys.add(inner_key)
                counter += 1
                sum_ = sum_ + inner_key
        prototype = sum_ / counter
        final_source_dict[source_dict[key].item()] = prototype

    target_dict = dict(zip(target, tar_labels))
    final_target_dict = {}
    processed_keys = set()
    for key in target_dict:
        if key in processed_keys:
            continue
        counter = 1
        sum_ = key
        for inner_key in target_dict:
            if not torch.all(torch.eq(key, inner_key)) and target_dict[key].item() == target_dict[inner_key].item():
                processed_keys.add(inner_key)
                counter += 1
                sum_ = sum_ + inner_key
        prototype = sum_ / counter
        final_target_dict[target_dict[key].item()] = prototype
    

    # Adding squared euclidean distances of prototypes of same classes. 
    # If a class is present in the source domain but not in the target domain
    # it is ignored
    sum_dists = 0

    for key in final_source_dict:
        if key in final_target_dict:
            s = ((final_source_dict[key] - final_target_dict[key]) ** 2).sum(axis=0)
            sum_dists = sum_dists + s
    
    return sum_dists

def class_alignment_loss2(source, target, src_labels, tar_labels):
    # Computing prototypes for each class as the mean of the extracted features
    source_dict = {}
    target_dict = {}
    source_dict = dict(zip(source, src_labels))
    target_dict = dict(zip(target, tar_labels))

    final_source_dict = {}
    final_target_dict = {}

    # print(f"source_dict = {source_dict}")

    # Compute prototypes for source domain
    for key, label in source_dict.items():
        if label.item() not in final_source_dict:
            final_source_dict[label.item()] = []
        final_source_dict[label.item()].append(key)

    # Compute prototypes for target domain
    for key, label in target_dict.items():
        if label.item() not in final_target_dict:
            final_target_dict[label.item()] = []
        final_target_dict[label.item()].append(key)

    # Calculate squared Euclidean distances of prototypes of same classes
    sum_dists = torch.tensor(0.).cuda()

    for label, source_prototypes in final_source_dict.items():
        if label in final_target_dict:
            target_prototypes = final_target_dict[label]
            source_prototypes = torch.mean(torch.stack(source_prototypes), dim=0, keepdim=True)
            target_prototypes = torch.mean(torch.stack(target_prototypes), dim=0, keepdim=True)
            dists = torch.sum((source_prototypes - target_prototypes) ** 2, dim=-1)
            sum_dists += dists.sum()

    return sum_dists

