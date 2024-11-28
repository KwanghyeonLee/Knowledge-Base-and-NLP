import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *
from datetime import datetime
import logging

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def make_logger(path, name):
    log_path = os.path.join(path, name)
    make_dir(log_path)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] - %(message)s")
    
    # StreamHandler : console Message
    stream_formatter = logging.Formatter('%(asctime)s: %(message)s', "%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    
    # FileHandler : file Message
    today = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(os.path.join(log_path, today + '.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def logger_result(opt, task_type, avg_losses, auc_scores, **kwargs):
    epoch = kwargs['epoch'] if 'epoch' in kwargs else None
    opt['logger'].info('===========================================')
    opt['logger'].info('Task: {}'.format(task_type))
    if epoch is not None:
        opt['logger'].info('Epoch: {}'.format(epoch))
    loss_dict_name2idx, loss_dict_idx2name = dictionary_for_loss(opt)
    # log losses
    opt['logger'].info('Losses:')
    for loss_name, loss_idx in loss_dict_name2idx.items():
        opt['logger'].info('{}: {:.4f}'.format(loss_name, avg_losses[loss_idx]))
    # log auc scores
    opt['logger'].info('AUC Scores:')
    if opt['model_name'] == 'STL':
        opt['logger'].info('AUC: {:.3f}'.format(auc_scores))
    else:
        opt['logger'].info('ARN: {:.3f}'.format(auc_scores[0]))
        opt['logger'].info('CMVr: {:.3f}'.format(auc_scores[1]))
    opt['logger'].info('===========================================')
        
def dictionary_for_loss(opt):
    if opt['model_name'] == 'STL':
        loss_dict_name2idx = {'loss_{}'.format(opt['class_type']): 0}
        loss_dict_idx2name = {0: 'loss'}
    elif opt['model_name'] == 'ADMTL':
        loss_dict_name2idx = {'total_loss': 0, 'task_loss': 1, 'adv_loss': 2, 'diff_loss': 3,
                              'ARN_loss': 4, 'CMVr_loss': 5, 'diff_ARN_loss': 6, 'diff_CMVr_loss': 7}
        loss_dict_idx2name = {0: 'total_loss', 1: 'task_loss', 2: 'adv_loss', 3: 'diff_loss',
                              4: 'ARN_loss', 5: 'CMVr_loss', 6: 'diff_ARN_loss', 7: 'diff_CMVr_loss'}
    else:
        loss_dict_name2idx = {'total_loss': 0, 'ARN_loss': 1, 'CMVr_loss': 2}
        loss_dict_idx2name = {0: 'total_loss', 1: 'ARN_loss', 2: 'CMVr_loss'}
    return loss_dict_name2idx, loss_dict_idx2name
    
def fix_random_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def plot_roc_curve(fpr, tpr, auc_score, plotter, save_path):
    figure = plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plotter.figure_plot('roc_curve', figure, 0)
    plt.close(figure)


def make_model_name_dir(default_path):
    model_name_list = ['STL', 'FSMTL', 'SPMTL', 'ADMTL']
    for model_name in model_name_list:
        if not os.path.exists(os.path.join(default_path, model_name)):
            os.mkdir(os.path.join(default_path, model_name))


def get_auc_score(pred, true):
    fpr, tpr, thresholds = roc_curve(true, pred)
    auc_score = auc(fpr, tpr)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return auc_score, best_threshold, [fpr, tpr, thresholds]


def get_precision_recall_ap(pred, true):
    precision, recall, thresholds = precision_recall_curve(true, pred)
    ap = average_precision_score(true, pred)
    return precision, recall, ap


def make_counterfactual_sample(batch, feature_idx_list, scale):
    '''
        batch : dataset batch (tensor), shape = (batch_size, feature_size)
        feature_idx_list : feature index that we want to add noise (list)
        scale : scale of noise (float)
    '''

    # make noise all of the input features have the same distribution (standard normal distribution)
    noise = np.random.normal(0, 1, batch.shape)
    sample = np.zeros_like(batch)

    # remove noise from features that we don't want to add noise but still remain the same shap
    sample[:, feature_idx_list] = noise[:, feature_idx_list]  # shape 뭐냐 이거

    # scale noise
    sample = sample * scale

    # sample
    sample = batch + torch.tensor(sample).float()

    return sample


def make_histogram_plot(result1, result2, save_path, bins=100):
    bin_size = 1/bins
    total_count = len(result1)
    figure = plt.figure(figsize=(7, 4))
    FSMTL_list, FSMTL_bins, FSMTL_patches = plt.hist(
        result2, bins=bins, label='FSMTL', alpha=0.5, color='mediumpurple')
    ADMTL_list, ADMTL_bins, ADMTL_patches = plt.hist(
        result1, bins=bins, label='ADMTL', color='mediumpurple')
    plt.legend(loc='upper right', fontsize='xx-large')
    plt.xlabel('DIF, bin size = {}'.format(bin_size))
    plt.ylabel('count(total= {})'.format(total_count))
    plt.savefig(save_path)
    plt.close(figure)
    return FSMTL_list, ADMTL_list
