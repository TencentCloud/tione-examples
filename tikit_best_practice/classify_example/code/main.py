import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.cuda import amp
from torch.cuda.amp import GradScaler
from tiacc_training import tiacc_training, tiacc_init

try:
    from tikit.client import Client
    client = Client("your_secret_id", "your_secret_key", "<region>")
except Exception as e:
    print("TIACC - run local!")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# This is passed in via launch.py
parser.add_argument("--local_rank", type=int, default=0)
# This needs to be explicitly passed in
parser.add_argument("--local_world_size", type=int, default=1)

parser.add_argument('--model_save_path', default='/opt/ml/model/', type=str,
                    help='model save path')
parser.add_argument('--tiacc', dest='tiacc', action='store_true',
                    help='use tiacc-training acceleration')

best_acc1 = 0

@tiacc_init(params={'training_framework': 'pytorch'})
def main():
    args = parser.parse_args()

    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"TIACC - [{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")

    print(
        f"TIACC - [{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    main_worker(args.local_world_size, args.local_rank, args)

    # Tear down the process group
    dist.destroy_process_group()


def main_worker(ngpus_per_node, local_rank, args):
    global best_acc1

    n = torch.cuda.device_count() // ngpus_per_node
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))
    args.gpu = device_ids[0]
    args.total_batch_size = dist.get_world_size() * args.batch_size

    print("TIACC - args.gpu", args.gpu, local_rank)
    print("TIACC - total_batch_size", args.total_batch_size)

    print(
        f"TIACC - [{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )
   
    # Data loading code
    #traindir = os.path.join(args.data, 'train/')
    #valdir = os.path.join(args.data, 'val/')
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    print("TIACC - traindir", traindir)
    print("TIACC - valdir", valdir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    num_classes = len(train_dataset.classes)
    num_classes_val = len(val_dataset.classes)
    print(f"TIACC - trainset classes num: {num_classes}")
    print(f"TIACC - trainset size: {len(train_dataset)}")
    print(f"TIACC - valset classes num: {num_classes_val}")
    print(f"TIACC - valset size: {len(val_dataset)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    # create model
    if args.arch == "resnest50":
        from resnest import resnest50
        model = resnest50()
    elif args.arch not in model_names:
        logging.error("not support this arch function, please choose one of {}".format(model_names))
        sys.exit(-1)
    elif args.pretrained:
        print("TIACC - using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        model = models.__dict__[args.arch]()

    #last fc
    num_fc_feat = model.fc.in_features
    model.fc = nn.Linear(num_fc_feat, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("TIACC - loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("TIACC - loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("TIACC - nlo checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [6, 8], 0.1)

    # add mixed precision scaler
    args.mixed_precision = args.tiacc
    if args.tiacc:
        schedulePolicy = "TimeSchedulePolicy"
        policy = tiacc_training.calc.MixedPrecision_TrainingPolicy(policy=schedulePolicy, 
                                                                   start_time=args.start_epoch,
                                                                   end_time=args.epochs)
        optimizer = tiacc_training.calc.get_fused_optimizer(optimizer)
        #policy = MixedPrecision_TrainingPolicy(policy=schedulePolicy, 
        #                                       start_time=0, end_time=3)

    scaler = torch.cuda.amp.GradScaler(
            init_scale=128,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=1000000000,
            enabled=args.mixed_precision,
        )

    for epoch in range(args.start_epoch, args.epochs+1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        scheduler.step()

        if args.tiacc:
            args.mixed_precision = policy.enable_mixed_precision(epoch, scaler=scaler)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scaler, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.model_save_path)


def train(train_loader, model, criterion, optimizer, epoch, scaler, args):
    # torch amp
    #scaler = amp.GradScaler()
    speed = AverageMeter('Speed', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])

    progress = ProgressMeter(
        len(train_loader),
        [speed, batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] LR: [{}]".format(epoch, lr_list[0]))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            images = images.float()

        with amp.autocast(enabled=args.mixed_precision):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        ## compute gradient and do SGD step
        optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        speed.update(args.total_batch_size / (time.time() - end))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank in [-1, 0]:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                images = images.float()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and args.local_rank in [-1, 0]:
                progress.display(i)

        try:
            client.push_training_metrics(int(time.time()), 
                                         {"acc1": float(format(top1.avg, '.3f')), 
                                          "acc5": float(format(top5.avg, '.3f'))}, 
                                         epoch=epoch)
        except Exception as e:
            pass

        # TODO: this should also be done with the ProgressMeter
        print('TIACC - * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Epoch={epoch}'
              .format(top1=top1, top5=top5, epoch=epoch))

    return top1.avg


def save_checkpoint(state, is_best, train_dir, filename='checkpoint.pth.tar'):
    torch.save(state, train_dir+filename)
    if is_best:
        shutil.copyfile(train_dir+filename, train_dir+'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
