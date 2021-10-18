from torch.autograd import Variable
import torch
import argparse
from dataset import dataset_cifar100
from utils import *
import os
from torch.optim.lr_scheduler import MultiStepLR
from biasloss import BiasLoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import densenet121, resnet18, shufflenetv2

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--base_lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=160, type=int, help='training epochs')
parser.add_argument('--pretrained', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--chkp_path', type=str, help='path to pretrained checkpoint')
parser.add_argument('--data_path', type=str, help='path to dataset')
parser.add_argument('--classes', default=100, type=int, help='number of classes')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--chkp_dir', default='./checkpoint', type=str, help='checkpoints directory')
parser.add_argument('--accuracies_dir', default='./accuracies', type=str, help='accuracies directory')
parser.add_argument('--milestones', default=[80, 122], type=int, nargs='+', help='milestones/epochs on which change lr')
parser.add_argument('--device', default=0, type=int, help='device id of GPU')
parser.add_argument('--model', default='resnet18', type=str, help='training model, options - resnet18, densenet121, shufflenetv2')
parser.add_argument('--norm_mode', default='global', type=str, help='mode for normalisation of the variance. Options: local, global')


args = parser.parse_args()

if not os.path.isdir(args.chkp_dir):
    os.mkdir(args.chkp_dir)

if not os.path.isdir(args.accuracies_dir):
    os.mkdir(args.accuracies_dir)


def run():
    writer = SummaryWriter()

    train_set, test_set = dataset_cifar100(batch_size=args.batch_size, data_path=args.data_path)
    accuracies, accuracies_train = np.zeros(args.epochs), np.zeros(args.epochs)

    criterion = BiasLoss(normalisation_mode=args.norm_mode)
    if args.model == 'resnet18':
        model = resnet18(num_classes=args.classes).to(args.device)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=args.classes).to(args.device)
    elif args.model == 'shufflenetv2':
        model = shufflenetv2(num_classes=args.classes).to(args.device)
    else:
        print('model choice is incorrect, please choose from the available options')
        exit()

    print('number of the parameters - ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    if args.pretrained:
        checkpoint = torch.load(args.chkp_path)
        model.load_state_dict(checkpoint['net'])

    for epoch in range(args.epochs):

        print('\nEpoch: %d' % epoch)
        model.train()
        test_loss, correct, total = 0, 0, 0

        for batch_idx, (input, target) in enumerate(train_set):
            if input.shape[0] != args.batch_size:
                continue
            it = epoch * len(train_set) + batch_idx

            optimizer.zero_grad()
            input, target = Variable(input.to(args.device)), Variable(target.to(args.device))
            output, features = model(input)

            loss = criterion(features=features, output=output, target=target)
            loss.backward()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if it % 25 == 0:
                writer.add_scalar('Loss/train', loss.detach().cpu().numpy(), it)

            progress_bar(batch_idx, len(train_set), 'Loss: %.9f' % (loss.item()))
            optimizer.step()

        scheduler.step()
        acc = 100. * correct / total
        writer.add_scalar('Accuracy/train', acc, epoch)

        accuracies_train[epoch] = acc
        np.savetxt(os.path.join(args.accuracies_dir, "accuracy_train.csv"), accuracies_train, delimiter=",", fmt='%s')

        test(model=model, testloader=test_set, epoch=epoch, device=args.device,
             accuracies=accuracies, writer=writer)


def test(epoch, model, testloader, device, accuracies, writer):
    global best_acc
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    accuracies[epoch] = acc
    np.savetxt(os.path.join(args.accuracies_dir, "accuracy.csv"), accuracies, delimiter=",", fmt='%s')
    state = {'net': model.state_dict(),'acc': acc,'epoch': epoch,}
    torch.save(state, os.path.join(args.chkp_dir, str(epoch) + 'ckpt.pth'))
    writer.add_scalar('Accuracy/test', acc, epoch)

    return acc


if __name__ == '__main__':
    run()