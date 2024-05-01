"""
BaseTrainer is used to show the training details without making our final trainer too complicated.
Users can extend this class to add more functionalities.
"""

from colorama import Fore, Style
from loguru import logger
from tabulate import tabulate


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.3f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


class BaseTrainer:
    def print_dataset_details(self):
        table = tabulate(
            [
                ["Train", len(self.train_loader.dataset)],
                ["Validation", len(self.val_loader.dataset)],
            ],
            headers=["Dataset", "Size"],
            tablefmt="fancy_grid",
        )
        logger.info("Dataset details:")
        print(f"{Fore.GREEN}", end="")
        print(table)
        print(f"{Style.RESET_ALL}", end="")

    def print_model_details(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        total_params = trainable_params + non_trainable_params
        table = tabulate(
            [
                ["Trainable", human_format(trainable_params)],
                ["Non-trainable", human_format(non_trainable_params)],
                ["Total", human_format(total_params)],
            ],
            headers=["Parameters", "Size"],
            tablefmt="fancy_grid",
        )
        logger.info("Model details:")
        print(f"{Fore.GREEN}", end="")
        print(table)
        print(f"{Style.RESET_ALL}", end="")

    def print_training_details(self):
        self.print_dataset_details()
        self.print_model_details()
