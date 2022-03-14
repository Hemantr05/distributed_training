# dream-systems
This project contains scripts/modules for distributed training on pytorch<br>
Based on current deep learning models, size of datasets, training methodologies; waiting for a model to train on a single GPU can be compared to waiting for an infant to take the first steps

Let's cut to the chase.<here>
In this repository I try to simplify the concepts and (a few) implementations for distributed training

## Introduction

There are generally two ways to distributed computation across multiple devices:

- Data Parallelism: where a single model gets replicated on multiple devices or multiple machines. Each of them processes different batches of data, then they merge their results. There exist many variants of this setup, that differ in how the different model replicas merge results, in whether they stay in sync at every batch or whether they are more loosely coupled, etc.
- Model Parallelism: where different parts of a single model run on different devices, processing a single batch of data together. This works best with models that have a naturally-parallel architecture, such as models that feature multiple branches.

Before going ahead there are two cases we need to think about first:

- Single-host (or node/machine), multi-device training
    - TF's [MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)
- Multi-worker distributed synchronous training or Multi-host, multi-device training 
    - TF's [MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy)
    - TF's [ParameterServerStrategy](https://www.tensorflow.org/guide/distributed_training#parameterserverstrategy)


Certain concepts and implementations have been picked up raw, you find the same [here](https://github.com/Hemantr05/dream-system#References) 

**References**

1. https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2#:~:text=PyTorch%20built%20two%20ways%20to,the%20network%20in%20multiple%20GPUs.
2. https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/
3. https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
4. https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
5. https://qr.ae/pGQvpr
6. https://keras.io/guides/distributed_training/
7. https://www.tensorflow.org/guide/distributed_training
8. https://lambdalabs.com/blog/introduction-multi-gpu-multi-node-distributed-training-nccl-2-0/
