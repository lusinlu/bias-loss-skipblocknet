import argparse
import logging
import torch
from collections import OrderedDict
from utils import load_checkpoint, accuracy, AverageMeter, data_loader

from skipblocknet import skipblocknet

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='SkipNet ImageNet Validation')
parser.add_argument('--data', required=True, type=str, help='path to ImageNet validation set')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
parser.add_argument('--num_classes', type=int, default=1000, help='Number classes in dataset')
parser.add_argument('--log_freq', default=10, type=int, help='batch logging frequency')
parser.add_argument('--checkpoint', default='skipblocknet-m.pth', type=str, help='path to the checkpoint ')


def validate(args):

    model = skipblocknet(num_classes=args.num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    load_checkpoint(model, args.checkpoint)

    param_count = sum([m.numel() for m in model.parameters()])
    logging.info('Model SkipNet created, param count: %d' % (param_count))


    loader = data_loader(args.data, args.batch_size)

    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(loader):
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)

            # measure and record accuracies
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))


            if i % args.log_freq == 0:
                logging.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Acc@1: {top1.val:>7.3f} '
                    'Acc@5: {top5.val:>7.3f} '.format(
                        i, len(loader), top1=top1, top5=top5))

    results = OrderedDict(
        top1=round(top1.avg, 4), top1_err=round(100 - top1.avg, 4),
        top5=round(top5.avg, 4), top5_err=round(100 - top5.avg, 4),
        param_count=round(param_count / 1e6, 2))

    logging.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
        results['top1'], results['top1_err'], results['top5'], results['top5_err']))



def main():
    console_handler = logging.StreamHandler()
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.INFO)
    args = parser.parse_args()

    validate(args)


if __name__ == '__main__':
    main()
