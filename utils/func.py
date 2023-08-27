import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pickle

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def write(self, txt):
        with open(self.log_path, 'a') as f:
            f.write(txt)
            f.write("\r\n")


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']

def get_mem():
    return '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

def get_next(dataloader, i):
    try:
        batch = next(i)
    except:
        trainloader_iter = iter(dataloader)
        batch = next(trainloader_iter)
    return batch


def get_train_val_loader(train_dataset, val_dataset, train_bs, val_bs, labeled_ratio, extraT1_ratio, train_split_path, work_dir):
    """
    训练数据分为三个部分：train_loader，train_loader_T1和 train_loader_remain
    train_loader: 同时有变化的标签和T1的语义标签
    train_loader_T1:没有变化的标签，只有T1的语义标签
    train_loader_remain:剩下的 既没有变化标签，也没有T1语义标签
    """
    num_workers = 2
    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)
    partial_size = int(labeled_ratio * train_dataset_size)
    print('partial size: ', partial_size)
    extraT1_num = int(extraT1_ratio * train_dataset_size)
    print('extra T1 label:', extraT1_num)

    if train_split_path:
        train_ids = pickle.load(open(train_split_path, 'rb'))

    else:
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)
        pickle.dump(train_ids, open(os.path.join(work_dir, 'train_split.pkl'), 'wb'))

    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    if extraT1_ratio == 1 - labeled_ratio:    # 剩下的T1 label全部用上
        train_T1_sampler = SubsetRandomSampler(train_ids[partial_size:])
        train_remain_sampler = SubsetRandomSampler(train_ids[partial_size:])
    elif extraT1_ratio == 0:     # 全部都没有T1 label       以上这两种情况 都只需要将dataset分为两个部分
        train_T1_sampler = SubsetRandomSampler(train_ids[partial_size:])
        train_remain_sampler = SubsetRandomSampler(train_ids[partial_size:])
    else:
        train_T1_sampler = SubsetRandomSampler(train_ids[partial_size:partial_size + extraT1_num])
        train_remain_sampler = SubsetRandomSampler(train_ids[partial_size + extraT1_num:])


    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_bs,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    train_loader_T1 = DataLoader(train_dataset,
                                 batch_size=train_bs,
                                 sampler=train_T1_sampler,
                                 num_workers=num_workers,
                                 pin_memory=True)

    train_loader_remain = DataLoader(train_dataset,
                                     batch_size=train_bs,
                                     sampler=train_remain_sampler,
                                     num_workers=num_workers,
                                     pin_memory=True)


    return train_loader, train_loader_remain, train_loader_T1, val_loader

def save_model(model, save_path, iteration, loss, metric):
    torch.save({
        'net': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'iteration': iteration,
        'loss': loss,
        'metric': metric
    }, save_path)

def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s