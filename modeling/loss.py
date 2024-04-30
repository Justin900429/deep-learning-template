import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets)


def build_loss(cfg) -> ClassificationLoss:
    return ClassificationLoss(label_smoothing=cfg.LOSS.LABEL_SMOOTHING)
