# Distributed Training
This project contains scripts/modules for distributed training<br>
Based on current deep learning models, size of datasets, training methodologies; waiting for a model to train on a single GPU can be compared to waiting for an infant to take the first steps

Let's cut to the chase.<br>
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



# Pytorch Distributed Training

Pytorch has two ways to split models and data across multiple GPUs: `nn.DataParallel` and `nn.DistributedDataParallel`

**nn.DataParallel:**

- `nn.DataParallel` uses one process to compute the model weights and then distribute them to each GPU during each batch, networking quickly becomes a bottle-neck and GPU utilization is often very low.
<br>
- Furthermore, `nn.DataParallel` requires that all the GPUs be on the same node and doesn't work with Apex for mixed-precision training


**nn.DistributedDataParallel:**

- `DistributedDataParallel` model reproduces the **model onto each GPU**
- Machine has one processes per GPU, each model is controlled by each process
- GPUs can all be on the same node(machine) or across multiple nodes.
- Only gradients are passed between the processes/GPUs
    - During training:
        - Each process loads its own mini-batch from disk and passes it to its GPU
        - Each GPU does its forward pass
        - The gradients from all gpus are all-reduced across the GPUs
        - (NOTE) Gradients for each layer do not depend on previoud layers, so the all-reduced is calculated concurrently with the backwards pass to further alleviate the networking bottleneck
        - At the end of the backwards pass, every node has the average gradients (ensuring the model weights stay synchronized)

## Implementation


**Driver script:**
On execution of main.py on either modules, the script launches a processs for every GPU.<br>
Each process needs to know which gpu to use and where it ranks amongst all the processes that are running

Parameters:
- nodes: The total number of nodes or machines to be used (currently testes on 1)
- gpus: The number of GPUs on each node
- nr: The rank of the current node (machine) within all the nodes [0 to nodes-1]

Calculated: 
- world_size: The total number of processes to run (gpus * nodes)


**Training:**

- Pytorch's `DistributedDataParallel` model reproduces the **model onto each GPU**
- `nn.utils.data.DistributedSampler` makes sure that each process gets a **different slice of training data** while loading it
    - As the Sampler already distributes data differently, we can assign `shuffle=False`


**Drawbacks:**
<br>
Here we try to train one model on multiple gpus, instead of one.<br>
As elegant and simple the approach seems, there are few pitfalls.<br><br>

Probelm: 
- main GPU running out of memory
    - As the first GPU will save all the different outputs for the different GPUs to calculate the loss, which inturn increase the memory usage

Solution:
- Reduce batch size
- Mixed precision training

Alternative to `nn.DistributeDataParallel` is Nvidia's `Apex`, for mixed precision

**Mixed Precision**: The use of lower-precision operations ( float16 and bfloat16 ) in a model during training to make it run faster and use less memory. Using mixed precision can improve performance by more than 3 times on modern GPUs and 60% on TPUs.[[1]](https://keras.io/api/mixed_precision/).
<br> 
The weight will remain at 32-bit, whereas other parameters like loss, gradients, etc will be computed at 16-bits. More of that [here](https://qr.ae/pGQvpr)


## Usage

- single node, multi-devices
<br>

```
    $ python main.py --nodes 1 --gpus 2 --epochs 5
```

- multi-node, multi-devices
<br>

```
    $ python main.py --nodes 2 --gpus 2 --epochs 5
```


# Distributed Training on Nvidia's Apex

## Concept

Nvidia's Apex is the best alterative to traditional distributed training for the following reasons
- No `GPU running out of memory` errors, due to [mixed precision training](https://keras.io/api/mixed_precision/)
- Apex's `apex.parallel.DistributedDataParallel` internally allows only one GPU per process
- Mixed-precision training required that the loss is [scaled](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/) in order to prevent the gradients from underflowing. **Apex does this automatically**


## Changes as compared to torch

- *Replace nn.DistributedDataParallel with apex.parallel.DistributedDataParallel*
<br>


- *Saving model/loading checkpoints*

    ```python
    # Save checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, 'amp_checkpoint.pt')
    ...


    # Restore
    model = ...
    optimizer = ...
    checkpoint = torch.load('amp_checkpoint.pt')

    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])

    # Continue training
    ...
    ```

## Usage

- single node, multi-devices
<br>

``` 
    $ python main.py --nodes 1 --gpus 2 --epochs 5
```

- multi-node, multi-devices
<br>

``` 
    $ python main.py --nodes 2 --gpus 2 --epochs 5
```


**References**

1. [How to scale training on multiple gpus](https://towardsdatascience.com/how-to-scale-training-on-multiple-gpus-dae1041f49d2#:~:text=PyTorch%20built%20two%20ways%20to,the%20network%20in%20multiple%20GPUs)
2. [mixed precision trianing deep neural networks by Nvidia](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/)
3. https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
4. https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
5. https://qr.ae/pGQvpr
6. https://keras.io/guides/distributed_training/
7. https://www.tensorflow.org/guide/distributed_training
8. https://lambdalabs.com/blog/introduction-multi-gpu-multi-node-distributed-training-nccl-2-0/
9. [Distributed data parallel training using Pytorch on AWS](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)
10. [Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)