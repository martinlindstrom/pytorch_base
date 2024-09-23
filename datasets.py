# Torch-related imports
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2

# Other imports

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    
    def __len__(self):
        return len(self.subset)

def get_train_val_splits(full_dataset, val_split, train_transform, val_transform):
    """
    Returns train/val splits of sizes according to val_split, with specified transforms.
    Input:
        - full_dataset: A torch.utils.data.Dataset
        - val_split: float in [0,1] indicating size of val split
        - train_transform: data transforms to apply to the train split
        - val_transform: data transforms to apply to the validation split
    Returns:
        Two datasets: train_set, val_set
    """
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [1.-val_split, val_split])
    train_set = SubsetWithTransform(train_subset, train_transform)
    val_set = SubsetWithTransform(val_subset, val_transform)
    return train_set, val_set

def get_MNIST(args, testset_only, valsplit=0.1):
    # Define augmentations
    train_augmentations = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(28),
        v2.RandomRotation(15),
        v2.ToDtype(torch.float32, scale=True)
    ])
    test_augmentations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    if testset_only:
        # Return only the testset
        test_dataset = datasets.MNIST(args.data_path, train=False, download=False, transform=test_augmentations)
        return None, None, test_dataset
    else:
        # Return everything
        train_dataset = datasets.MNIST(args.data_path, train=True, download=False, transform=None) #IMPORTANT; add different transforms to train/val later
        test_dataset = datasets.MNIST(args.data_path, train=False, download=False, transform=test_augmentations)
        # Get random train/val splits with correct augmentations
        train_dataset, val_dataset = get_train_val_splits(train_dataset, valsplit, train_augmentations, test_augmentations)
        # Return
        return train_dataset, val_dataset, test_dataset

def get_CIFAR10(args, testset_only, valsplit=0.1):
    # Pre-calculated means and stds
    cifar10_mean = [0.49139965, 0.48215827, 0.44653103]
    cifar10_std = [0.24703230, 0.24348512, 0.26158816]
    # Define standard augmentations
    train_augmentations = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(size=32, padding=4),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(15),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    test_augmentations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    if testset_only:
        # Return only the restset 
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=False, transform=test_augmentations)
        None, None, test_dataset
    else:
        # Return everything
        train_dataset = datasets.CIFAR10(args.data_path, train=True, download=False, transform=None) #IMPORTANT; add different transforms to train/val later
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=False, transform=test_augmentations)
        # Get random train/val splits with correct augmentations
        train_dataset, val_dataset = get_train_val_splits(train_dataset, valsplit, train_augmentations, test_augmentations)
        # Return
        return train_dataset, val_dataset, test_dataset

def get_CIFAR100(args, testset_only, valsplit=0.1):
    # Pre-calculated means and stds
    cifar100_mean = [0.50707483, 0.48654887, 0.44091749]
    cifar100_std = [0.26733416, 0.25643855, 0.27615049]
    # Define standard augmentations
    train_augmentations = v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(15),
        v2.RandomResizedCrop(32),
        v2.RandomHorizontalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    test_augmentations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])
    if testset_only:
        # Return only the testset
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=False, transform=test_augmentations)
        return None, None, test_dataset
    else:
        # Return everything
        train_dataset = datasets.CIFAR10(args.data_path, train=True, download=False, transform=None) #IMPORTANT; add different transforms to train/val later
        test_dataset = datasets.CIFAR10(args.data_path, train=False, download=False, transform=test_augmentations)
        # Get random train/val splits with correct augmentations
        train_dataset, val_dataset = get_train_val_splits(train_dataset, valsplit, train_augmentations, test_augmentations)
        # Return
        return train_dataset, val_dataset, test_dataset

def get_imagenet(args, testset_only, valsplit=0.1):
    # Inspiration taken from PyTorch example here: 
    # https://github.com/pytorch/examples/blob/a308b4e97459b07c1b356642b2a8b4206c6d6de1/imagenet/main.py#L236
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    # Define standard augmentations
    train_augmentations = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    test_augmentations = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    if testset_only:
        # Return only the testset
        test_dataset = datasets.ImageNet(args.data_path, split="val", transform=test_augmentations)
        return None, None, test_dataset
    else:
        # Return everything
        train_dataset = datasets.ImageNet(args.data_path, split="train", transform=None)
        test_dataset = datasets.ImageNet(args.data_path, split="val", transform=test_augmentations)
        # Get random train/val splits with correct augmentations
        train_dataset, val_dataset = get_train_val_splits(train_dataset, valsplit, train_augmentations, test_augmentations)
        return train_dataset, val_dataset, test_dataset