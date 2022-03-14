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