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
                criterion1, criterion_consistency,
                optimizer, metric, lambda_unlabeled, device="cuda"):

        self.num_classes = num_classes
        self.net = net


        self.criterion1 = criterion1
        self.criterion_consistency = criterion_consistency
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.lambda_unlabeled = lambda_unlabeled

        self._to_device()

    def _to_device(self):
        self.net.to(self.device)
        self.criterion1.to(self.device)
        self.criterion_consistency.to(self.device)
        self.metric.to(self.device)

    def run(self, show_interval, train_loader, train_loader_T1, train_loader_remain, mode="semi", flag=0):
        if mode=="semi":
            print(('\n' + '%9s' * 8) % ("train", 'gpu', 'loss1_', 'lossCU', 'precision', 'recall', 'f1', 'iou'))
        else:
            print(('\n' + '%9s' * 10) % ("train", 'gpu', 'loss1', 'loss2', 'lossChg', 'loss1_', 'precision', 'recall', 'f1', 'iou'))

        train_remain_iter = iter(train_loader_remain)
        train_T1_iter = iter(train_loader_T1)
        train_iter = iter(train_loader)

        self.net.train()

        # loss and metrics
        loss1_labeled_meter = AverageValueMeter()
        loss2_labeled_meter = AverageValueMeter()
        losschg_labeled_meter = AverageValueMeter()
        loss1_labeled_new_meter = AverageValueMeter()

        loss1_unlabeled_meter = AverageValueMeter()
        loss1_unlabeled_meter_branch2 = AverageValueMeter()
        loss_CU_meter = AverageValueMeter()
        loss_CL_meter = AverageValueMeter()
        # loss_discriminator_meter = AverageValueMeter()

        cm_meter = CMMeter()

        pbar = tqdm(range(show_interval))
        for _ in pbar:
            # for param in self.discriminator.parameters():
            #     param.requires_grad = False

            # step 1: train net with labeled images
            img1, img2, label1, label2, label = get_next(train_loader, train_iter)
            img1, img2, label1, label2, label = img1.to(self.device), img2.to(self.device), \
                                                label1.to(self.device), label2.to(self.device), label.to(self.device)

            Seg1, Seg2, Chg, Seg1_ = self.net(img1, img2)

            loss1_labeled = self.criterion1(Seg1, label1)
            loss2_labeled = self.criterion1(Seg2, label2)
            loss_chg_labeld = self.criterion1(Chg, label)
            loss1_labeled_new = self.criterion1(Seg1_, label1)


            loss1_labeled_value = loss1_labeled.cpu().detach().numpy()
            loss2_labeled_value = loss2_labeled.cpu().detach().numpy()
            loss_chg_labeled_value = loss_chg_labeld.cpu().detach().numpy()
            loss1_labeled_new_value = loss1_labeled_new.cpu().detach().numpy()

            loss1_labeled_meter.add(loss1_labeled_value)
            loss2_labeled_meter.add(loss2_labeled_value)
            losschg_labeled_meter.add(loss_chg_labeled_value)
            loss1_labeled_new_meter.add(loss1_labeled_new_value)

            metrics = self.metric(Chg, label)  # 只列出了S(t1,t2)的预测结果
            cm_meter.add(metrics)
            precision, recall, f1, iou = cm_meter.get_metrics()

            # step 2: consistency loss with unlabeled images

            if mode == "semi":
                if flag == 0: # 全部的T1 label
                    # loss_CL
                    # pred1_labeled_softmax = F.softmax(pred1_labeled, dim=1)
                    # pred2_labeled_softmax = F.softmax(pred2_labeled, dim=1)
                    # loss_CL = self.criterion_CL(pred1_labeled_softmax, pred2_labeled_softmax)
                    # loss_CL_value = loss_CL.cpu().detach().numpy()
                    # loss_CL_meter.add(loss_CL_value)

                    img1, img2, label1, _, _ = get_next(train_loader_T1, train_T1_iter)
                    img1, img2, label1 = img1.to(self.device), img2.to(self.device), label1.to(self.device)

                    Seg1, Seg2, Chg, Seg1_ = self.net(img1, img2)

                    # loss_CU
                    pred1_g_softmax = F.softmax(Seg1, dim=1)
                    pred2_g_softmax = F.softmax(Seg1_, dim=1)
                    # loss_unlabeled = self.criterion_unlabel(pred1_g, pred2_g)
                    loss_CU = self.criterion_consistency(pred1_g_softmax, pred2_g_softmax)
                    loss_CU_value = loss_CU.cpu().detach().numpy()
                    loss_CU_meter.add(loss_CU_value)

                    # loss seg1
                    loss1_unlabeled = self.criterion1(Seg1, label1)
                    loss1_unlabeled_value = loss1_unlabeled.cpu().detach().numpy()
                    loss1_unlabeled_meter.add(loss1_unlabeled_value)

                    # loss seg1_
                    loss1_unlabeled_branch2 = self.criterion1(Seg1_, label1)
                    loss1_unlabeled_branch2_value = loss1_unlabeled_branch2.cpu().detach().numpy()
                    loss1_unlabeled_meter_branch2.add(loss1_unlabeled_branch2_value)


                    # loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld + loss1_labeled_new +\
                    #          self.lambda_unlabeled * (loss1_unlabeled + loss1_unlabeled_branch2 + loss_CU)

                    loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld + loss1_labeled_new + \
                             self.lambda_unlabeled * (loss1_unlabeled + loss1_unlabeled_branch2)    # 去掉loss_CU

                elif flag == 1:   # partly have T1 label, partly dont
                    # loss for data with T1 label
                    img1, img2, label1, _, _ = get_next(train_loader_T1, train_T1_iter)
                    img1, img2, label1 = img1.to(self.device), img2.to(self.device), label1.to(self.device)

                    Seg1, Seg2, Chg, Seg1_ = self.net(img1, img2)
                    # pred1_g_softmax = F.softmax(Seg1, dim=1)
                    # pred2_g_softmax = F.softmax(Seg1_, dim=1)
                    # loss_CU = self.criterion_consistency(pred1_g_softmax, pred2_g_softmax)
                    loss1_unlabeled = self.criterion1(Seg1, label1)
                    loss1_unlabeled_branch2 = self.criterion1(Seg1_, label1)
                    loss1_unlabeled_branch2_value = loss1_unlabeled_branch2.cpu().detach().numpy()
                    loss1_unlabeled_meter_branch2.add(loss1_unlabeled_branch2_value)

                    # loss for data without T1 label
                    img1, img2, _, _, _ = get_next(train_loader_remain, train_remain_iter)
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    Seg1, Seg2, Chg, Seg1_ = self.net(img1, img2)
                    pred1_g_softmax = F.softmax(Seg1, dim=1)
                    pred2_g_softmax = F.softmax(Seg1_, dim=1)
                    loss_CU = self.criterion_consistency(pred1_g_softmax, pred2_g_softmax)
                    loss_CU_value = loss_CU.cpu().detach().numpy()
                    loss_CU_meter.add(loss_CU_value)

                    loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld + loss1_labeled_new + \
                        self.lambda_unlabeled * (loss1_unlabeled + loss1_unlabeled_branch2) + \
                        self.lambda_unlabeled * loss_CU

                elif flag == 2:
                    # loss for data without T1 label
                    img1, img2, _, _, _ = get_next(train_loader_remain, train_remain_iter)
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    Seg1, Seg2, Chg, Seg1_ = self.net(img1, img2)
                    pred1_g_softmax = F.softmax(Seg1, dim=1)
                    pred2_g_softmax = F.softmax(Seg1_, dim=1)
                    loss_CU = self.criterion_consistency(pred1_g_softmax, pred2_g_softmax)
                    loss_CU_value = loss_CU.cpu().detach().numpy()
                    loss_CU_meter.add(loss_CU_value)

                    loss1_unlabeled_branch2_value = 0

                    loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld + loss1_labeled_new + \
                             self.lambda_unlabeled * loss_CU

                pbar.set_description(
                    ('%9s' * 2 + '%9.4g' * 6) % (
                    "train", get_mem(), loss1_unlabeled_branch2_value, loss_CU_value, precision, recall, f1, iou))

            elif mode == "supervised":
                loss_S = loss1_labeled + loss2_labeled + loss_chg_labeld + loss1_labeled_new

                pbar.set_description(
                    ('%9s' * 2 + '%9.4g' * 8) % (
                    "train", get_mem(), loss1_labeled_value, loss2_labeled_value, loss_chg_labeled_value, loss1_labeled_new_value,
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
            'loss1_new': loss1_labeled_new_meter.mean,
            'loss_CU': loss_CU_meter.mean,
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

        # 测试模式
        self.net.eval()

        # loss和指标
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











