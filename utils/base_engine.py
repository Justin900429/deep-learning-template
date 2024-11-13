"""
BaseTrainer is used to show the training details without making our final trainer too complicated.
Users can extend this class to add more functionalities.
"""

import os

import accelerate
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from yacs.config import CfgNode as CN

from config import show_config
from utils.meter import AverageMeter


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.3f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class BaseEngine:
    def __init__(self, accelerator: accelerate.Accelerator, cfg: CN):
        # Setup accelerator for distributed training (or single GPU) automatically
        self.base_dir = os.path.join(cfg.LOG_DIR, cfg.PROJECT_DIR)
        self.accelerator = accelerator

        if self.accelerator.is_main_process:
            os.makedirs(self.base_dir, exist_ok=True)
            show_config(cfg)
        self.accelerator.wait_for_everyone()

        self.cfg = cfg
        self.device = self.accelerator.device

        self.sub_task_progress = Progress(
            TextColumn("{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=True,
            disable=not self.accelerator.is_main_process,
        )
        self.epoch_progress = Progress(
            *self.sub_task_progress.columns,
            TextColumn("| [bold blue]best acc: {task.fields[acc]:.3f}"),
            transient=True,
            disable=not self.accelerator.is_main_process,
        )
        self.live_process = Live(Group(self.epoch_progress, self.sub_task_progress))
        self.live_process.start(refresh=self.live_process._renderable is not None)

        # Monitor for the time
        self.iter_time = AverageMeter()
        self.data_time = AverageMeter()

    def print_dataset_details(self):
        self.accelerator.print(
            "📁 \033[1mLength of dataset\033[0m:\n"
            f" - 💪 Train: {len(self.train_loader.dataset)}\n"
            f" - 📝 Validation: {len(self.val_loader.dataset)}"
        )

    def print_model_details(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params
        self.accelerator.print(
            "🤖 \033[1mModel Parameters:\033[0m\n"
            f" - 🔥 Trainable: {trainable_params}\n"
            f" - 🧊 Non-trainable: {non_trainable_params}\n"
            f" - 🤯 Total: {total_params}"
        )

    def print_training_details(self):
        try:
            self.print_dataset_details()
        except Exception:
            pass
        try:
            self.print_model_details()
        except Exception:
            pass

    def reset(self):
        self.data_time.reset()
        self.iter_time.reset()

    def close(self):
        self.live_process.stop()
        self.accelerator.end_training()
