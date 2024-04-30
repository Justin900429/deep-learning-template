"""
This folder define the model and the loss function for training.
Please feel free to add more files or modules to suit your need
"""

from .loss import build_loss
from .model import build_model

__all__ = ["build_model", "build_loss"]
