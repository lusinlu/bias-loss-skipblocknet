import argparse
import torch
from dataset import dataset_cifar100
from models import densenet121, resnet18, shufflenetv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--device', default=1, type=int, help='device id of GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('--classes', default=100, type=int, help='number of classes')
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='testing model, options - resnet18, densenet121, shufflenetv2')

    args = parser.parse_args()

    if args.model == 'resnet18':
        model = resnet18(num_classes=args.classes).to(args.device)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=args.classes).to(args.device)
    elif args.model == 'shufflenetv2':
        model = shufflenetv2(num_classes=args.classes).to(args.device)
    else:
        print('model choice is incorrect, please choose from the available options')
        exit()

    _, test_set = dataset_cifar100(batch_size=args.batch_size, data_path=args.data_path)

    model.load_state_dict(torch.load(args.checkpoint)['net'])
    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_set):
            print("iteration: {}\ttotal {} iterations".format(batch_idx + 1, len(test_set)))

            image, label = image.to(args.device), label.to(args.device)

            output, _ = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 accuracy: ", correct_1 / len(test_set.dataset) * 100.)
    print("Top 5 accuracy: ", correct_5 / len(test_set.dataset) * 100.)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))