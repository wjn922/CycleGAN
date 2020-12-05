import argparse
import os
import random
import shutil
import time
import warnings
import tqdm

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, models
from PIL import Image

from evaluation.dataset import ImagenetDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
    parser.add_argument("--datapath", type=str, default="./data/horse2zebra/testA",
                        help="path to datasets. (default:./data/horse2zebra/testA)")
    parser.add_argument("--label", type=str, default="horse",
                        help="dataset label", choices=['horse','zebra','apple','orange'])
    parser.add_argument("--cuda", action="store_true", help="Enables cuda")
    parser.add_argument("--image-size", type=int, default=256,
                        help="size of the data crop (squared assumed). (default:256)")
    parser.add_argument("--manualSeed", type=int, default=None,
                        help="Seed for initializing training. (default:none)")
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-p", "--print_freq", default=100, type=int,
                        metavar="N", help="print frequency. (default:100)")


    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = ImagenetDataset(root=args.datapath, label=args.label, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load('./evaluation/resnet50-19c8e357.pth'))
    model.to(device)

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # evaluate
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(dataloader):
            if args.cuda is not None:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = model(images)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print('\n * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
	main()