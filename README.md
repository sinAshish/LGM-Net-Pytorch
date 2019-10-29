# LGM-Net-Pytorch
A pytorch implementation of LGM-Net: Learning to Generate Matching Networks for Few-Shot Learning(ICML 2019).


## Abstract

In this work, we propose a novel meta-learning approach for few-shot classification, which learns transferable prior knowledge across tasks and directly produces network parameters for similar unseen tasks with training samples. Our approach, called LGM-Net, includes two key modules, namely, TargetNet and MetaNet. The TargetNet module is a neural network for solving a specific task and the MetaNet module aims at learning to generate functional weights for TargetNet by observing training samples. We also present an intertask normalization strategy for the training process to leverage common information shared across different tasks. The experimental results on Omniglot and miniImageNet datasets demonstrate that LGM-Net can effectively adapt to similar unseen tasks and achieve competitive performance, and the results on synthetic datasets show that transferable prior knowledge is learned by the MetaNet module via mapping training data to functional weights. LGM-Net enables fast learning and adaptation since no further tuning steps are required compared to other meta-learning approaches.

### Usage

`python train.py --ways 5 --shot 1 --lr 1e-3 --epochs 100 `
`python train.py --ways 20 --shot 1 --lr 1e-3 --epochs 10`
`python train.py --ways 5 --shot 5 --lr 1e-3 -e 10`
`python train.py --ways 20 --shot 5 --lr 1e-2 -e 10`

### Results

The dataset used in this implementation is CIFAR-100 (FC100). I don't have any GPUs, so I had to test my model with a fairly smaller dataset compared to the authors, <br>

The model performs at par +- 3% with the author's implementation on 5 way 1 shot, 20 way 1 shot, and 5 way 5 shot. I could not test the 20 way 5 shot, because of my laptop's constraints. Loading the dataset on Colab from drive is slower than my laptop. <br>

Hope you find this implementation useful.

### Acknowledgements

The dataloader is borrowed from the [Author's code](https://github.com/likesiwell/LGM-Net/).
I'll add the dataloader in Pytorch, cause the one by the author's is very slow.For now, the model works fine on MacBook Pro 8GB version.

## ToDo
- [x] Implement the model
- [x] Prepare the dataset
- [x] Train the model
- [x] Check the results

```
If you find this code useful in your research, please consider citing the paper by the original authors.
```
