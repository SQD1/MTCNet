import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.cuda import amp
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
from utils.metrics import CMMeter
from utils.func import get_next, get_mem


class TrainEpoch:
    def __init__(self, num_classes, net,
                criterion1, optimizer, metric, device="cuda"):

        self.num_classes = num_classes
        self.net = net
        self.criterion1 = criterion1
        self.optimizer = optimizer
        self.metric = metric
        self.device = device


        self._to_device()

    def _to_device(self):
        self.net.to(self.device)
        self.criterion1.to(self.device)
        self.metric.to(self.device)

    def run(self, show_interval, train_loader):
        print(('\n' + '%9s' * 9) % ("train", 'gpu', 'loss1', 'loss2', 'lossChg', 'precision', 'recall', 'f1', 'iou'))

        train_iter = iter(train_loader)


        # 训练模式
        self.net.train()

        # loss和指标
        loss1_labeled_meter = AverageValueMeter()
        loss2_labeled_meter = AverageValueMeter()
        losschg_labeled_meter = AverageValueMeter()
        # loss1_labeled_new_meter = AverageValueMeter()

        cm_meter = CMMeter()

        pbar = tqdm(range(show_interval))
        for _ in pbar:
            # for param in self.discriminator.parameters():
            #     param.requires_grad = False

            # step 1: train net with labeled images
            img1, img2, label1, label2, label = get_next(train_loader, train_iter)
            img1, img2, label1, label2, label = img1.to(self.device), img2.to(self.device), \
                                                label1.to(self.device), label2.to(self.device), label.to(self.device)

            Seg1, Seg2, Chg, _ = self.net(img1, img2)

            loss1_labeled = self.criterion1(Seg1, label1)
            loss2_labeled = self.criterion1(Seg2, label2)
            loss_chg_labeld = self.criterion1(Chg, label)
            # loss1_labeled_new = self.criterion1(Seg1_, label1)


            loss1_labeled_value = loss1_labeled.cpu().detach().numpy()
            loss2_labeled_value = loss2_labeled.cpu().detach().numpy()
            loss_chg_labeled_value = loss_chg_labeld.cpu().detach().numpy()
            # loss1_labeled_new_value = loss1_labeled_new.cpu().detach().numpy()

            loss1_labeled_meter.add(loss1_labeled_value)
            loss2_labeled_meter.add(loss2_labeled_value)
            losschg_labeled_meter.add(loss_chg_labeled_value)
            # loss1_labeled_new_meter.add(loss1_labeled_new_value)

            metrics = self.metric(Chg, label)  # 只列出了S(t1,t2)的预测结果
            cm_meter.add(metrics)
            precision, recall, f1, iou = cm_meter.get_metrics()


            # loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld
            loss_S = loss_chg_labeld             # single task


            pbar.set_description(
                ('%9s' * 2 + '%9.4g' * 7) % (
                "train", get_mem(), loss1_labeled_value, loss2_labeled_value, loss_chg_labeled_value,
                precision, recall, f1, iou))



            # loss_unlabeled_weighted = self.lambda_unlabeled * loss_unlabeled
            self.optimizer.zero_grad()
            loss_S.backward()
            self.optimizer.step()


        precision, recall, f1, iou = cm_meter.get_metrics()

        logs = {
            'loss1': loss1_labeled_meter.mean,
            'loss2': loss2_labeled_meter.mean,
            'lossChg': losschg_labeled_meter.mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }

        return logs


class ValEpoch:
    def __init__(self, num_classes, net,
                criterion1,
                metric, device="cuda"):

        self.num_classes = num_classes
        self.net = net
        self.criterion = criterion1
        # self.criterion_consistency = criterion_consistency
        self.metric = metric
        self.device = device

        self._to_device()

    def _to_device(self):
        self.net.to(self.device)
        self.criterion.to(self.device)
        self.metric.to(self.device)

    @torch.no_grad()
    def run(self, dataloader):
        print(('\n' + '%10s' * 7) % ("val", 'gpu', 'loss', 'precision', 'recall', 'f1', 'iou'))

        self.net.eval()

        # loss and metrics
        loss_meter = AverageValueMeter()
        cm_meter = CMMeter()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, (img1, img2, _, _, label) in pbar:
            # x = x.to(self.device)
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            _, _, Chg, _ = self.net(img1, img2)
            loss = self.criterion(Chg, label)
            loss_labeled_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_labeled_value)
            metrics = self.metric(Chg, label)
            cm_meter.add(metrics)
            precision, recall, f1, iou = cm_meter.get_metrics()

            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % ("val", get_mem(), loss_labeled_value,
                                                                precision, recall, f1, iou))
        precision, recall, f1, iou = cm_meter.get_metrics()
        logs = {
            'loss': loss_meter.mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        }
        return logs