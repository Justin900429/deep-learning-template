from typing import Tuple

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    """Please define your own `Dataset` here. We provide an example for CIFAR-10 dataset."""

    pass


def get_loader(cfg) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.DATA.ROOT, train=True, download=True, transform=train_transform
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=cfg.DATA.ROOT, train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.EVAL.NUM_WORKERS,
        batch_size=cfg.EVAL.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader
