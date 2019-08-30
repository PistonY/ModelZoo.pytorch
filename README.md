# ModelZoo for Pytorch

This is a model zoo project under Pytorch. In this repo I will implement some of basic classification 
models which have good performance on ImageNet. Then I will train them in most fair way as possible and
try my best to get SOTA model on ImageNet. In this repo I'll only consider FP16.


## Usage
### Environment
- OS: Ubuntu 18.04
- CUDA: 10.0, CuDNN: 7.5
- Devices: I use 8 * RTX 2080ti(8 * V100 should be much better /cry). This project is under FP16, it's recommend to use FP16 friendly devices like 
RTX series, V100. If you want to totally reproduce my research, you'd better use same (total) batch size with me.

### Requirement
- Pytorch: >= 1.1.0
- [Apex](https://github.com/NVIDIA/apex): nightly version. Support optimized FP16 tools. 
- [TorchToolbox](https://github.com/deeplearningforfun/torch-toolbox): nightly version. 
Helper functions to make your code simpler and more readable, it's a optional tools
if you don't want to use it just write them yourself.


## Baseline models

|model | epochs| dtype |batch size*|gpus  | lr  |  tricks|speed|memory cost(MiB)^|top1/top5 |
|:----:|:-----:|:-----:|:---------:|:----:|:---:|:------:|:---:|:--------------:|:---------:|
|resnet50|120  |FP16   |128        |  8   |0.4  | -      | -   |   7700         |77.36/-    |
|resnet50v2|120  |FP16 |128        |  8   |0.4  | -      | -   |   7700         |77.06/93.44|

- I use nesterov SGD and cosine lr decay with 5 warmup epochs by default[2][3] (to save time), it's more common and effective.
- *Batch size is pre GPU holds. Total batch size should be (batch size * gpus).
- ^This is average memory cost.
- Resnet 50 top5 in log file is not right(actually is top -5), just ignore it.

## Ablation Study on Tricks
Here are lots of tricks to improve accuracy during this years.(If you have another idea please open an issue.)
I want to verify them in a fair way.


Tricks: RandomRotation, OctConv[14], Drop out, Label Smoothing[4], Sync BN, SwitchNorm[6], Mixup[17], no decay bias[7], 
Cutout[5], swish activation[10], Stochastic Depth[9], Lookahead Optimizer[11], Pre-active(ResnetV2)[12], 
DCNv2[13], LIP[16].


Special: Zero-initialize the last BN, just call it 'Zero γ'.

I'll only use 120 epochs and 128*8 batch size to train them.
I know some tricks may need train more time or larger batch size but it's not fair for others.
You can think of it as a performance in the current situation.


|model | epochs| dtype |batch size*|gpus  | lr  |  tricks|degree|top1/top5  |improve |
|:----:|:-----:|:-----:|:---------:|:----:|:---:|:------:|:----:|:---------:|:------:|
|resnet50|120  |FP16   |128        | 8    |0.4  | -      |   -  |77.36/-    |baseline|
|resnet50|120  |FP16   |128        | 8    |0.4  |Label smoothing|smoothing=0.1|77.78/93.80 |+0.42 |
|resnet50|120  |FP16   |128        | 8    |0.4  |No decay bias  |-            |77.28/93.61*|-0.08 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Sync BN        |-            |77.31/93.49^|-0.05 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Mixup          |alpha=0.2    |77.49/93.73 |+0.13 |
|resnet50|120  |FP16   |128        | 8    |0.4  |RandomRotation |degree=15    |76.64/93.28 |-1.15 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Cutout         |read code    |77.44/93.62 |+0.08 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Dropout        |rate=0.3     |77.11/93.58 |-0.25 |
|resnet50|120  |FP16   |128        | 8    |0.4  |Lookahead-SGD  |    -        |77.23/93.39 |-0.13 |
|resnet50v2|120  |FP16 |128        | 8    |0.4  |pre-active     |  -          |77.06/93.44 |-0.30 |


- *If you only have 1k(128 * 8) batch size, it's not recommend to use this which made unstable convergence and finally 
    can't get a higher accuracy.Original paper use 64k batch size but impossible for me to follow.
- ^Though Sync BN didn't improve any accuracy, it's a magic experience which looks like using one GPU to train.


## ToDo
- [ ] Resume training
- [ ] Try Nvidia-DALI
- [ ] Multi-node(distributed) training by Apex or BytePS
- [ ] I may try AutoAugment.This project aims to train models by ourselves to observe and learn,
     it's impossible for me to train this, just copy feels meaningless.

## Citation
```
@misc{ModelZoo.pytorch,
  title = {Basic deep conv neural network reproduce and explore},
  author = {X.Yang},
  URL = {},
  year = {2019}
  }
```

## Reference
- [1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [2] [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf)
- [3] [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
- [4] [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
- [5] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
- [6] [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/pdf/1806.10779.pdf) [OpenSourse](https://github.com/switchablenorms/Switchable-Normalization)
- [7] [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes](https://arxiv.org/pdf/1807.11205.pdf)
- [8] [MIXED PRECISION TRAINING](https://arxiv.org/pdf/1710.03740.pdf)
- [9] [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
- [10] [SEARCHING FOR ACTIVATION FUNCTIONS](https://arxiv.org/pdf/1710.05941.pdf)
- [11] [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
- [12] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)
- [13] [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/pdf/1811.11168.pdf)
- [14] [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)
- [15] [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
- [16] [LIP: Local Importance-based Pooling](https://arxiv.org/pdf/1908.04156v1.pdf)
- [17] [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)