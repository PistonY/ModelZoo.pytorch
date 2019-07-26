# ModelZoo for Pytorch

This is a model zoo project under Pytorch. In this repo I will implement some of basic classification 
models which have good performance on ImageNet. Then I will train them in most fair way as possible and
try my best to get SOTA model on ImageNet. In this repo I'll only consider pure FP16.

## Baseline models

|model | epochs| dtype |batch size*|gpus  | lr  |  tricks|speed|memory cost(MiB)^|top1/top5|proposed top1|
|:----:|:-----:|:-----:|:---------:|:----:|:---:|:------:|:---:|:--------------:|:-------:|:-----------:|
|resnet50|120  |FP16   |128        |  8   |0.4  | -      | 950 |   7700         |77.35/-  |77.15(FP32)  |

    - I use nesterov SGD and cosine lr decay with 5 warmup epochs by default[2](to save time), it's more common and effective.
    - *Batch size is pre GPU holds. Total batch size should be (batch size * gpus).
    - ^This is average memory cost.
    - Resnet 50 top5 in log file is not right(actually is top -5), just ignore it.

## Ablation Study on Tricks
Here are lots of tricks to improve accuracy during this years.(If you have another idea please open an issue.)
I want to verify them in a fair way.


Tricks: Drop out, Label Smoothing[4], Sync BN, SwitchNorm[6], Mixup[5], no bias decay[7], Cutout[5], 
swish activation[10], Stochastic Depth[9]

Special: Zero-initialize the last BN, just call it 'Zero γ'.

I'll only use 120 epochs and 128*8 batch size to train them,
I know some tricks may need train more time or larger batch size but it's not fair for others.
You can think of it as a performance in the current situation.

|model | epochs| dtype |batch size*|gpus  | lr  |  tricks|top1/top5  |improve |
|:----:|:-----:|:-----:|:---------:|:----:|:---:|:------:|:---------:|:------:|
|resnet50|120  |FP16   |128        | 8    |0.4  | -      |77.35/-    |baseline|
|resnet50|120  |FP16   |128        | 8    |0.4  |Label smoothing|77.78/93.80 |+0.43 |
|resnet50|120  |FP16   |128        | 8    |0.4  |No bias decay  |77.28/93.61*|-0.07 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Sync BN        |77.31/93.49^|-0.04 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Mixup          | | |

    - *If you only have 1k(128*8) batch size, it's not recommend to use this which made unstable convergence and finally 
    can't get a higher accuracy.Origin paper use 64k batch size but impossible for me to follow.
    - ^Though Sync BN didn't improve any accuracy, it's a magic experience which looks like you are using one GPU to train.
      


##ToDo
    - Try Nvidia-Dali
    - Multi-node(distributed) training by Apex or BytePS
    - I may try AutoAugment.This project aim to train models by ourselves to observe and learn,
     it's impossible for me to train this, just copy feels meaningless.
## Citation
```
@misc{ModelZoo.pytorch,
  title = {ModelZoo for Pytorch: Basic classification model reproduce and explore},
  author = {X.Yang},
  URL = {},
  year = {2019} 
  }
```
## Reference
    - [1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    - [2] [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
    - [3] [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)
    - [4] [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
    - [5] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
    - [6] [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf) [OpenSourse](https://github.com/switchablenorms/Switchable-Normalization)
    - [7] [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/pdf/1807.11205.pdf)
    - [8] [MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)
    - [9] [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
    - [10] [SEARCHING FOR ACTIVATION FUNCTIONS](https://arxiv.org/pdf/1710.05941.pdf)
    