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

- `configs/`: Configuration files for setting up different aspects of your models and training environments.
- `dataset/`: Modules to handle data loading and preprocessing.
- `modeling/`: Definition and instantiation of the neural network models and loss function.
- `utils/`: Utilities for various tasks like logging and performance metrics.
- `train.py`: The main script to start the training process.

## ⚙️ Config Setup

Configure your models and training setups with ease. Modify the `config.py` file to suit your experimental needs. Our system uses [YACS](https://github.com/rbgirshick/yacs), which allows for a hierarchical configuration with overrides for command-line options.

## 🏋️‍♂️ Training

### Basic Usage

To start a training, run:

```shell
python train.py --config configs/your_config.yaml

# Concrete example
python traing.py --config configs/cifar/cifar-small.yaml
```

### Override the config with command line

Users can override the options with the `--opts` flag. For instance, to resume the training:

```shell
python train.py --config configs/your_config.yaml --opts TRAIN.RESUME_CHECKPOINT path/to/checkpoint

# Concrete example
python train.py --config configs/cifar/cifar-small.yaml --opts TRAIN.RESUME_CHECKPOINT logs/cifar-small/checkpoint/best_model_epoch_10.pth
```

Please check the config setup section for more details.

### Multi-GPU Training

This project template is made based on [accelerate](https://github.com/huggingface/accelerate) to provide multi-GPU training. A simple example to train a model with 2 GPUs:

```shell
accelerate launch --multi_gpu --num_processes=2 train.py --config configs/your_config.yaml --opts (optional)

# Concrete example
accelerate launch --multi_gpu --num_processes=2 train.py --config configs/cifar/cifar-small.yaml \
    --opts TRAIN.RESUME_CHECKPOINT logs/cifar-small/checkpoint/best_model_epoch_10.pth
```

## 🛠 How to Add Your Code?

1. **Integrating New Models:** Place your model files in the `modeling/` folder and update the configurations accordingly.
2. **Adding New Datasets:** Implement data handling in the `dataset/` folder and reference it in your config files.
3. **Utility Scripts:** Enhance functionality by adding utility scripts in the `utils/` folder.
4. **Customized Training Process**: Please change the `train.py` to modify the training process.

## 🙌 Special Thanks

Special thanks to the creators of [accelerate](https://github.com/huggingface/accelerate) and [YACS](https://github.com/rbgirshick/yacs), whose tools have significantly enhanced the flexibility and usability of this template. Also, we appreciate the inspiration from existing projects like those by [L1aoXingyu](https://github.com/L1aoXingyu/Deep-Learning-Project-Template) and [victoresque](https://github.com/victoresque/pytorch-template).

Feel free to modify and adapt this README to better fit the specifics and details of your project.
