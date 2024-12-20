import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ListDataset
from model import CrowdCounterModel
from utils import save_checkpoint

parser = argparse.ArgumentParser(description='Crowd Counting')

parser.add_argument('train_json', metavar='TRAIN', help='path to train json')
parser.add_argument('test_json', metavar='TEST', help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str, help='path to the pretrained model')
parser.add_argument('task', metavar='TASK', type=str, help='task id to use.')
parser.add_argument('--max_train_samples', type=int, default=None,
                    help='maximum number of samples to train on in each epoch')


def main():
    global args, best_prec1

    best_prec1 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size = 1
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 400
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = int(time.time())

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    torch.cuda.manual_seed(args.seed)

    # Check for GPU availability and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CrowdCounterModel().to(device)  # Move model to device (GPU or CPU)
    criterion = nn.MSELoss(reduction='sum').to(device)  # Move criterion to device

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre, map_location=device)  # Ensure checkpoint is loaded to the correct device
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_list, model, criterion, optimizer, epoch, device)
        prec1 = validate(val_list, model, device)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)


def train(train_list, model, criterion, optimizer, epoch, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    data_set = ListDataset(train_list, shuffle=True, transform=transform, train=True, seen=model.seen,
                           batch_size=args.batch_size,
                           num_workers=args.workers)

    train_loader = DataLoader(data_set, batch_size=args.batch_size)

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):

        img = img.to(device)  # Move img to the device (GPU/CPU)
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)  # Move target to the device
        target = Variable(target)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_list, model, device):
    print('Begin validation')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    dataset = ListDataset(val_list, shuffle=False, transform=transform, train=False)

    test_loader = DataLoader(dataset, batch_size=args.batch_size)

    model.eval()
    mae = 0

    for i, (img, target) in enumerate(tqdm(test_loader, desc="Validating")):
        img = img.to(device)  # Move img to the device (GPU/CPU)
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).to(device))  # Move target to device

    mae = mae / len(test_loader)
    print(f' * MAE {mae:.3f} ')

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr

    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


if __name__ == '__main__':
    main()