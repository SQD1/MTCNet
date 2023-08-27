import argparse
from dataset_WHU import WHUDataset
from utils.func import get_train_val_loader, Logger, get_learning_rate, save_model, format_logs
from utils.metrics import *
from utils.epoch import TrainEpoch, ValEpoch
from model.networks import ChgSegNet_V2
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="train process")
    parser.add_argument("--work_dirs", type=str, default='../semi_checkpoints')
    parser.add_argument("--log", type=str, default='semi_0.1_lambda_1.0_allT1')
    parser.add_argument('--ratio', '--labeled_ratio', type=float, default=0.1)
    parser.add_argument('--T1_ratio', '--T1_labeled_ratio', type=float, default=0.9)   # extra T1 labels
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Base learning rate for training with polynomial decay.")   # 2e-4
    parser.add_argument("--num-steps", type=int, default=24000,
                        help="Number of iterations.")
    parser.add_argument('--si', '--show_interval', type=int, default=200)
    parser.add_argument('--epoch_start_unsup', type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="train dataset batch size.")
    parser.add_argument("--val-batch-size", type=int, default=2,
                        help="val dataset batch size.")
    parser.add_argument("--data_parallel", action='store_true', default=True)
    parser.add_argument("--lambda_unlabeled", type=float, default=1.0)
    parser.add_argument("--train_split_path", type=str, default=None)
    # 第一次运行train.py时，令train_split_path=None, 会在work_dirs目录下自动生成一个train_split.pkl文件，表示数据集的一个随机序列
    # 后续运行train.py时，将train_split_path设置为文件train_split.pkl的路径，有标签/无标签数据的划分都将基于该pkl文件。

    return parser.parse_args()



def train():
    num_classes = 2
    args = get_arguments()

    # logger
    save_dir = os.path.join(args.work_dirs, args.log)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, "train.log"))
    logger.write(str(args))

    # dataset and dataloader
    train_dataset = WHUDataset("/data/sqd/WHU256/train")    # label [B, H, W]
    val_dataset = WHUDataset("/data/sqd/WHU256/test")

    loaders = get_train_val_loader(train_dataset, val_dataset, args.batch_size, args.val_batch_size, args.ratio, args.T1_ratio,
                                   train_split_path=args.train_split_path,
                                   work_dir=args.work_dirs)

    train_loader, train_loader_remain, train_loader_T1, val_loader = loaders

    # model
    net = ChgSegNet_V2(2)

    # load checkpoints
    # checkpoint = torch.load("/home/peter/sqd/semi_checkpoints/WHU3/semi_0.1_lambda_1.0_noextraT1/best.pth")
    # net.load_state_dict(checkpoint['net'])
    # print("load completed!")

    if args.data_parallel:
        net = nn.DataParallel(net)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion_CE = nn.CrossEntropyLoss()
    # criterion_CE = nn.BCELoss()
    criterion_consistency = nn.MSELoss()

    metric = ChangeMetrics(False)

    train_runner = TrainEpoch(num_classes, net,
                              criterion_CE, criterion_consistency,
                              optimizer, metric, lambda_unlabeled=args.lambda_unlabeled, device="cuda")
    val_runner = ValEpoch(num_classes, net, criterion_CE, metric)

    best_val_metric = 0
    bms = 0

    # flag 0: Use all the remaining T1 labels
    # flag 1: Part with T1 label，and part without T1 label
    # flag 2: No extra T1 label
    if args.ratio + args.T1_ratio == 1:
        flag = 0
    elif args.T1_ratio == 0:
        flag = 2
    else:
        flag = 1

    for i in range(0, args.num_steps, args.si):
        print('Iteration: %d - lr: %.5f' % (i, get_learning_rate(optimizer)))
        if i < args.epoch_start_unsup * args.si:
            train_log = train_runner.run(args.si,
                                         train_loader, train_loader_remain, train_loader_T1,
                                         "supervised", flag)
        else:
            train_log = train_runner.run(args.si,
                                         train_loader, train_loader_remain, train_loader_T1,
                                         "semi", flag)
        val_log = val_runner.run(val_loader)
        val_metric = val_log['f1']

        save_model(net, os.path.join(save_dir, 'latest.pth'), i, val_log['loss'], val_metric)

        if val_log['f1'] > best_val_metric:
            best_val_metric = val_log['f1']
            bms = val_log
            save_model(net, os.path.join(save_dir, 'best.pth'), i, val_log['loss'], val_metric)

        logger.write('Epoch:\t' + str(i))
        logger.write('Train:\t' + format_logs(train_log))
        logger.write('Val:\t' + format_logs(val_log))
        logger.write("Best:\t" + format_logs(bms))
        logger.write("\n")

        print("train:", train_log)
        print("val:", val_log)
        print("best_metric:\t" + format_logs(bms))


if __name__ == "__main__":
    train()









