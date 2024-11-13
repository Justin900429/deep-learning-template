import argparse
import json
import os
import time

import accelerate
import torch
from yacs.config import CfgNode as CN

from config import config_to_str, create_cfg, merge_possible_with_base
from dataset import get_loader
from modeling import build_loss, build_model
from utils.base_engine import BaseEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


class Engine(BaseEngine):
    def __init__(self, accelerator: accelerate.Accelerator, cfg: CN):
        super().__init__(accelerator, cfg)

        # Setup model, loss, optimizer, and dataloaders
        model = build_model(cfg)
        self.loss_fn = build_loss(cfg)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.TRAIN.LR * self.accelerator.num_processes,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )

        with self.accelerator.main_process_first():
            train_loader, val_loader = get_loader(cfg)

        # Prepare model, optimizer, loss_fn, and dataloaders for distributed training (or single GPU)
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader)
        self.min_loss = float("inf")
        self.current_epoch = 1

        self.max_acc = 0

        # Resume or not
        if self.cfg.MODEL.RESUME_CHECKPOINT is not None:
            with self.accelerator.main_process_first():
                self.load_from_checkpoint()

    def load_from_checkpoint(self):
        """
        Load model and optimizer from checkpoint for resuming training.
        Modify this for custom components if needed.
        """
        checkpoint = self.cfg.MODEL.RESUME_CHECKPOINT
        if not os.path.exists(checkpoint):
            self.accelerator.print(f"[WARN] Checkpoint {checkpoint} not found. Skipping...")
            return
        self.accelerator.load_state(checkpoint)

        if not os.path.exists(os.path.join(checkpoint, "meta_data.json")):
            self.accelerator.print(
                f"[WARN] meta data for resuming training is not found in {checkpoint}. Skipping..."
            )
            return

        with open(os.path.join(checkpoint, "meta_data.json"), "r") as f:
            meta_data = json.load(f)
        self.current_epoch = meta_data.get("epoch", 0) + 1
        self.max_acc = meta_data.get("max_acc", 0)
        self.accelerator.print(
            f"[WARN] Checkpoint loaded from {self.cfg.MODEL.RESUME_CHECKPOINT}, continue training or validate..."
        )
        del checkpoint

    def save_checkpoint(self, save_path: str):
        self.accelerator.save_state(save_path)
        with open(os.path.join(save_path, "meta_data.json"), "w") as f:
            json.dump(
                {
                    "epoch": self.current_epoch,
                    "max_acc": self.max_acc,
                },
                f,
            )

    def _train_one_epoch(self):
        epoch_progress = self.sub_task_progress.add_task("loader", total=len(self.train_loader))
        self.model.train()
        step_loss = 0
        start = time.time()
        for loader_idx, (img, label) in enumerate(self.train_loader, 1):
            current_step = (self.current_epoch - 1) * len(self.train_loader) + loader_idx
            self.data_time.update(time.time() - start)
            with self.accelerator.accumulate(self.model):
                output = self.model(img)
                loss = self.loss_fn(output, label)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = self.accelerator.gather(loss.detach().cpu().clone()).mean()
                step_loss += loss.item() / self.cfg.TRAIN.ACCUM_ITER
            self.iter_time.update(time.time() - start)

            if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                self.accelerator.log(
                    {
                        "loss/train": step_loss,
                    },
                    step=current_step,
                )
                step_loss = 0

            self.accelerator.log(
                {
                    "time/iter": self.iter_time.val,
                    "time/data": self.data_time.val,
                },
                step=current_step,
            )
            self.sub_task_progress.update(epoch_progress, advance=1)

            start = time.time()
        self.sub_task_progress.remove_task(epoch_progress)

    def validate(self):
        valid_progress = self.sub_task_progress.add_task("validate", total=len(self.val_loader))
        total_acc = 0
        self.model.eval()
        for img, label in self.val_loader:
            pred = self.model(img)
            batch_pred, batch_label = self.accelerator.gather_for_metrics((pred, label))
            correct = (batch_pred.argmax(1) == batch_label).sum().item()
            total_acc += correct / len(label)
            self.sub_task_progress.update(valid_progress, advance=1)
        total_acc /= len(self.val_loader)
        if self.accelerator.is_main_process:
            self.accelerator.print(f"val. acc. at epoch {self.current_epoch}: {total_acc:.3f}")
            self.accelerator.log(
                {
                    "acc/val": total_acc,
                },
                step=(self.current_epoch - 1) * len(self.train_loader),  # Use train steps
            )
        if self.accelerator.is_main_process and total_acc > self.max_acc:
            save_path = os.path.join(self.base_dir, "checkpoint")
            self.accelerator.print(f"new best found with: {total_acc:.3f}, save to {save_path}")
            self.max_acc = total_acc
            self.save_checkpoint(
                os.path.join(
                    save_path,
                    f"epoch_{self.current_epoch}",
                ),
            )
        self.sub_task_progress.remove_task(valid_progress)

    def setup_training(self):
        os.makedirs(os.path.join(self.base_dir, "checkpoint"), exist_ok=True)
        self.accelerator.init_trackers(
            self.accelerator.project_configuration.project_dir, config=config_to_str(cfg)
        )

    def train(self):
        train_progress = self.epoch_progress.add_task(
            "Epoch",
            total=self.cfg.TRAIN.EPOCHS,
            completed=self.current_epoch - 1,
            acc=self.max_acc,
        )
        if self.accelerator.is_main_process:
            self.print_training_details()
            self.setup_training()
        self.accelerator.wait_for_everyone()
        for epoch in range(self.current_epoch, self.cfg.TRAIN.EPOCHS + 1):
            self.current_epoch = epoch
            self._train_one_epoch()
            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                self.accelerator.wait_for_everyone()
                self.validate()
            self.epoch_progress.update(train_progress, advance=1, acc=self.max_acc)
        self.epoch_progress.stop_task(train_progress)

    def reset(self):
        super().reset()
        self.max_acc = 0


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    if args.config:
        merge_possible_with_base(cfg, args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)

    project_config = accelerate.utils.ProjectConfiguration(
        project_dir=cfg.PROJECT_DIR,
        logging_dir=cfg.LOG_DIR,
    )
    accelerator = accelerate.Accelerator(
        log_with=cfg.PROJECT_LOG_WITH,
        project_config=project_config,
        gradient_accumulation_steps=cfg.TRAIN.ACCUM_ITER,
        mixed_precision=cfg.TRAIN.MIXED_PRECISION,
    )
    # Set seed for reproducibility
    accelerate.utils.set_seed(args.seed, device_specific=True)
    engine = Engine(accelerator, cfg)
    engine.train()
    engine.close()
