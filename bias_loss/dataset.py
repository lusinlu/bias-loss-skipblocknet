from torchvision import datasets, transforms
import torch
import torch.utils.data as data

transform_train_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

transform_test_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])


def dataset_cifar100(batch_size, data_path):

    trainset_cifar100 = datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transform_train_cifar100)
    trainloader_cifar100 = torch.utils.data.DataLoader(
        trainset_cifar100, batch_size=batch_size, shuffle=True, num_workers=2)

    testset_cifar100 = datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test_cifar100)
    testloader_cifar100 = torch.utils.data.DataLoader(
        testset_cifar100, batch_size=100, shuffle=False, num_workers=2)

    return trainloader_cifar100, testloader_cifar100




























