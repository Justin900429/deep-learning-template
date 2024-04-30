import argparse
import datetime
import os
import time

import accelerate
import torch
from loguru import logger
from yacs.config import CfgNode as CN

from config import config_to_str, create_cfg, merge_possible_with_base, show_config
from dataset import get_loader
from modeling import build_loss, build_model
from utils.meter import AverageMeter, MetricMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


class Trainer:
    def __init__(self, cfg: CN):
        os.makedirs(cfg.PROJECT_DIR, exist_ok=True)

        # Setup accelerator for distributed training (or single GPU) automatically
        config = accelerate.utils.ProjectConfiguration(
            project_dir=cfg.PROJECT_DIR,
            logging_dir="logs",
        )
        self.accelerator = accelerate.Accelerator(
            log_with=cfg.PROJECT_LOG_WITH, project_config=config
        )
        if self.accelerator.is_main_process:
            show_config(cfg)
            self.accelerator.init_trackers(cfg.PROJECT_DIR, config=config_to_str(cfg))
            logger.add(os.path.join("logs", cfg.PROJECT_DIR, "train.log"))
        self.cfg = cfg
        self.device = self.accelerator.device

        # Setup model, loss, optimizer, and dataloaders
        model = build_model(cfg)
        loss_fn = build_loss(cfg)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

        with self.accelerator.main_process_first():
            train_loader, val_loader = get_loader(cfg)

        # Prepare model, optimizer, loss_fn, and dataloaders for distributed training (or single GPU)
        (
            self.model,
            self.optimizer,
            self.loss_fn,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            model, optimizer, loss_fn, train_loader, val_loader
        )
        self.min_loss = float("inf")
        self.current_epoch = 0

        # Monitor for the time and the loss
        self.loss_meter = MetricMeter()
        self.iter_time = AverageMeter()
        self.data_time = AverageMeter()
        self.max_acc = 0

        # Resume or not
        if self.cfg.TRAIN.RESUME_CHECKPOINT is not None:
            with self.accelerator.main_process_first():
                self.load_from_checkpoint()

    def load_from_checkpoint(self):
        """
        Load model and optimizer from checkpoint for resuming training.
        Modify this for custom components if needed.
        """
        checkpoint = self.cfg.TRAIN.RESUME_CHECKPOINT
        if not os.path.exists(checkpoint):
            logger.warning(f"Checkpoint {checkpoint} not found. Skipping...")
            return
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.optimizer.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["epoch"] + 1
        if self.accelerator.is_main_process:
            logger.info(
                f"Checkpoint loaded from {self.cfg.MODEL.CHECKPOINT}, continue training or validate..."
            )
        del checkpoint

    def _train_one_epoch(self):
        self.model.train()

        start = time.time()
        for loader_idx, (img, label) in enumerate(self.train_loader, 1):
            self.data_time.update(time.time() - start)
            with self.accelerator.accumulate(self.model):
                output = self.model(img)
                loss = self.loss_fn(output, label)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.iter_time.update(time.time() - start)

            if self.accelerator.is_main_process:
                self.loss_meter.update({"loss": loss.item()})

            if self.accelerator.is_main_process and (
                (loader_idx % self.cfg.TRAIN.LOG_EVERY_STEP == 0)
                or (loader_idx == len(self.train_loader))
            ):
                nb_future_iters = (cfg.TRAIN.EPOCHS - (self.current_epoch + 1)) * len(
                    self.train_loader
                ) - (loader_idx)
                eta_seconds = self.iter_time.avg * nb_future_iters
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    f"Epoch [{self.current_epoch}/{self.cfg.TRAIN.EPOCHS}]({loader_idx}/{len(self.train_loader)}) | "
                    f"iter_time: {self.iter_time.avg:.3f}s | "
                    f"data_time: {self.data_time.avg:.3f}s | "
                    f"{self.loss_meter} | "
                    f"eta: {eta_str}"
                )
                self.accelerator.log(
                    {
                        "train_loss": loss.item(),
                    },
                    step=(self.current_epoch - 1) * len(self.train_loader) + loader_idx,
                )
            start = time.time()

    def validate(self):
        total_acc = 0
        self.model.eval()
        for loader_idx, (img, label) in enumerate(self.val_loader, 1):
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            if self.accelerator.is_main_process and (
                (loader_idx % self.cfg.EVAL.LOG_EVERY_STEP == 0)
                or (loader_idx == len(self.val_loader))
            ):
                logger.info(
                    f"Step [{loader_idx}/{len(self.val_loader)}] | Accuracy: {correct / len(label) * 100:.2f}%"
                )
            total_acc += correct / len(label)
        total_acc /= len(self.val_loader)
        if self.accelerator.is_main_process:
            logger.info(f"Validation accuracy: {total_acc:.3f}")
            self.accelerator.log(
                {
                    "val_acc": total_acc,
                },
                step=(self.current_epoch - 1) * len(self.val_loader)
                + len(self.val_loader),
            )
        if self.accelerator.is_main_process and total_acc > self.max_acc:
            save_path = os.path.join("logs", self.cfg.PROJECT_DIR, "checkpoint")
            os.makedirs(save_path, exist_ok=True)
            logger.info(
                f"New best model found with accuracy: {total_acc:.3f}, save to best_model_epoch_{self.current_epoch}.pth"
            )
            self.max_acc = total_acc
            torch.save(
                {
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "optimizer": self.optimizer.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                },
                os.path.join(
                    save_path,
                    f"best_model_epoch_{self.current_epoch}.pth",
                ),
            )

    def train(self):
        for epoch in range(1, self.cfg.TRAIN.EPOCHS + 1):
            self.current_epoch = epoch
            self._train_one_epoch()
            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                self.accelerator.wait_for_everyone()
                self.validate()

    def reset(self):
        self.max_acc = 0
        self.loss_meter.reset()
        self.data_time.reset()
        self.iter_time.reset()

    def close(self):
        self.accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config:
        merge_possible_with_base(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    trainer = Trainer(cfg)
    trainer.train()
    trainer.close()
