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

## References

- [Distributed data parallel training using Pytorch on AWS](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)
- [Distributed data parallel training in Pytorch](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)