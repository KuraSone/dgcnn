"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function

import argparse
import os
import shutil
import sys

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ModelNet40
from model import PointNet, DGCNN
from util import cal_loss, IOStream


def _init_() -> None:
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(f'checkpoints/{args.exp_name}'):
        os.makedirs(f'checkpoints/{args.exp_name}')
    if not os.path.exists(f'checkpoints/{args.exp_name}/models'):
        os.makedirs(f'checkpoints/{args.exp_name}/models')
    shutil.copy('main.py', f'checkpoints/{args.exp_name}/main.py.backup')
    shutil.copy('model.py', f'checkpoints/{args.exp_name}/model.py.backup')
    shutil.copy('util.py', f'checkpoints/{args.exp_name}/util.py.backup')
    shutil.copy('dataset.py', f'checkpoints/{args.exp_name}/dataset.py.backup')


def train(args_: argparse.Namespace, io_: IOStream) -> None:
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args_.num_points), num_workers=4,
                              batch_size=args_.batch_size, shuffle=True, drop_last=True,
                              persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args_.num_points), num_workers=4,
                             batch_size=args_.test_batch_size, shuffle=True, drop_last=False,
                             persistent_workers=True, pin_memory=True)

    device = torch.device("cuda" if args_.cuda else "cpu")

    # Try to load models
    model: nn.Module
    if args_.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args_.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt: optim.Optimizer
    if args_.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args_.lr * 100, momentum=args_.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args_.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args_.epochs, eta_min=args_.lr)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args_.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred_list = []
        train_true_list = []
        for data, label in tqdm(train_loader, desc=f'Train epoch {epoch}', file=sys.stdout):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            torch.autograd.backward(loss)
            opt.step()
            scheduler.step()
            predictions = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true_list.append(label.cpu().numpy())
            train_pred_list.append(predictions.detach().cpu().numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        output_str = (f'Train {epoch}, loss: {train_loss * 1.0 / count:.6f}, train acc: '
                      f'{metrics.accuracy_score(train_true, train_pred):.6f}, train avg acc: '
                      f'{metrics.balanced_accuracy_score(train_true, train_pred):.6f}')
        io_.cprint(output_str)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred_list = []
        test_true_list = []
        for data, label in tqdm(test_loader, desc=f'Test epoch {epoch}', file=sys.stdout):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            predictions = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true_list.append(label.cpu().numpy())
            test_pred_list.append(predictions.detach().cpu().numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        output_str = (f'Test {epoch}, loss: {test_loss * 1.0 / count:.6f}, test acc: {test_acc:.6f}, '
                      f'test avg acc: {avg_per_class_acc:.6f}')
        io_.cprint(output_str)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'checkpoints/{args_.exp_name}/models/model.t7')


def test(args_: argparse.Namespace, io_: IOStream) -> None:
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args_.num_points),
                             num_workers=8, batch_size=args_.test_batch_size, shuffle=True, drop_last=False,
                             persistent_workers=True, pin_memory=True)

    device = torch.device("cuda" if args_.cuda else "cpu")

    # Try to load models
    model: nn.Module = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args_.model_path))
    model = model.eval()
    test_true_list = []
    test_pred_list = []
    for data, label in tqdm(test_loader, desc='Test', file=sys.stdout):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        predictions = logits.max(dim=1)[1]
        test_true_list.append(label.cpu().numpy())
        test_pred_list.append(predictions.detach().cpu().numpy())
    test_true = np.concatenate(test_true_list)
    test_pred = np.concatenate(test_pred_list)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    output_str = f'Test :: test acc: {test_acc:.6f}, test avg acc: {avg_per_class_acc:.6f}'
    io_.cprint(output_str)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream(f'checkpoints/{args.exp_name}/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            f'Using GPU : {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
