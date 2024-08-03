# Deep Learning Project's Template

## 🚀 Introduction

Welcome to our Deep Learning Project Template, crafted for researchers and developers working with PyTorch. This template is designed to streamline the setup, execution, and modification of deep learning experiments, allowing you to focus more on model development and less on boilerplate code.

## ✨ Features

1. **Multi-GPU Support:** Utilize the power of multiple GPUs or devices to accelerate your training using [accelerate](https://github.com/huggingface/accelerate).
2. **Flexible Configuration:** Easily configure your experiments with the versatile [YACS](https://github.com/rbgirshick/yacs) configuration system, enabling quick adjustments for different scenarios.
3. **Clear Architecture:** Our template is structured for clarity and ease of use, ensuring you can understand and modify the code with minimal effort.
4. **Transparent Training Process:** Enjoy a clear display of the training process, helping you monitor performance and make necessary tweaks in real-time.

## 📂 Folder Structure

Our project is organized as follows to help you navigate and manage the codebase effectively:

```plaintext
📦deep-learning-template
 ├── 📂configs                # Configuration files for experiments
 │   ├── 📂cifar
 │   │   ├── cifar_big.yaml   # Configuration for a larger model
 │   │   └── cifar_small.yaml # Configuration for a smaller model
 │   └── 📄default.yaml       # Default config for all experiemnts
 ├── 📂dataset                # Modules for data handling
 │   └── 📄data_loader.py     # Data loader script
 ├── 📂modeling               # Neural network models and loss functions
 │   └── 📄model.py           # Example model file
 ├── 📂utils                  # Utility scripts for various tasks
 │   ├── 📄base_trainer.py    # Base Trainer class for printing training details
 │   ├── 📄logger.py          # Logging utilities
 │   └── 📄metrics.py         # Performance metrics
 ├── 📄.gitignore             # Specifies intentionally untracked files to ignore
 ├── 📄LICENSE                # License file for the project
 ├── 📄README.md              # README file with project details
 ├── 📄config.py              # Main configuration script
 ├── 📄linter.sh              # Shell script for formating the code
 ├── 📄requirements.txt       # Dependencies and libraries
 └── 📄engine.py              # Main training and validation script
```

## ⚙️ Configuration

Configure your models and training setups with ease. Modify the `config.py` file to suit your experimental needs. Our system uses [YACS](https://github.com/rbgirshick/yacs), which allows for a hierarchical configuration with overrides for command-line options. The recommeneded structure we used:

```python
# Basic setup of the project
cfg = CN()
cfg._BASE_ = None
cfg.PROJECT_DIR = None
cfg.PROJECT_LOG_WITH = ["tensorboard"]

# Control the modeling settings
cfg.MODEL = CN()
# ...

# Control the loss settings
cfg.LOSS = CN()
# ...

# Control the dataset settings (e.g., path)
cfg.DATA = CN()
# ...

# Control the training setup (e.g., lr, epoch)
cfg.TRAIN = CN()
# ...

# Control the training setup (e.g., batch size)
cfg.EVAL = CN()
# ...
```

## 🏋️‍♂️ Training

### Basic Usage

To start a training, run:

```shell
python engine.py --config configs/your_config.yaml

# Concrete example
python traing.py --config configs/cifar/cifar-small.yaml
```

After the training start, users can find the training folder called `logs`. To modify the default setting, please change the option `LOG_DIR`. Followed by `logs` is the `PROJECT_DIR` defined in the config file.

```
📦{LOG_DIR}/{PROJECT_DIR}
 ├── 📂checkpoint           # Folder for saving checkpoints 
 ├── 📂...                  # Other files setup by tracker(s)
 └── 📄train.log            # Logs during training 
```

### Override the config with command line

Users can override the options with the `--opts` flag. For instance, to resume the training:

```shell
python engine.py --config configs/your_config.yaml --opts TRAIN.RESUME_CHECKPOINT path/to/checkpoint

# Concrete example
python engine.py --config configs/cifar/cifar-small.yaml --opts TRAIN.RESUME_CHECKPOINT logs/cifar-small/checkpoint/best_model_epoch_10.pth
```

Please check the config setup section for more details.

### Multi-GPU Training

This project template is made based on [accelerate](https://github.com/huggingface/accelerate) to provide multi-GPU training. A simple example to train a model with 2 GPUs:

```shell
accelerate launch --multi_gpu --num_processes=2 engine.py --config configs/your_config.yaml --opts (optional)

# Concrete example
accelerate launch --multi_gpu --num_processes=2 engine.py --config configs/cifar/cifar-small.yaml \
    --opts TRAIN.RESUME_CHECKPOINT logs/cifar-small/checkpoint/best_model_epoch_10.pth
```

### Tracker

Trackers such as `tensorboard`, `wandb`, and `aim` can be setup from the `PROJECT_LOG_WITH` option. We support multiple trackers at once through accelerate! Users are encouraged to find our which is the best for the project from [here](https://huggingface.co/docs/accelerate/usage_guides/tracking). Below are some examples to open the local monitor:

```shell
# tensorboard
tensorboard --logdir logs

# aim
aim up --repo logs
```

## 🛠 How to Add Your Code?

1. **Integrating New Models:** Place your model files in the `modeling/` folder and update the configurations accordingly.
2. **Adding New Datasets:** Implement data handling in the `dataset/` folder and reference it in your config files.
3. **Utility Scripts:** Enhance functionality by adding utility scripts in the `utils/` folder.
4. **Customized Training Process**: Please change the `engine.py` to modify the training process.

## TODO

- [ ] Support iteration based training with infinite loader.

## 🙌 Special Thanks

Special thanks to the creators of [accelerate](https://github.com/huggingface/accelerate) and [YACS](https://github.com/rbgirshick/yacs), whose tools have significantly enhanced the flexibility and usability of this template. Also, we appreciate the inspiration from existing projects like those by [L1aoXingyu](https://github.com/L1aoXingyu/Deep-Learning-Project-Template) and [victoresque](https://github.com/victoresque/pytorch-template).

Feel free to modify and adapt this README to better fit the specifics and details of your project.
