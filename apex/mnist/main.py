import os
import argparse
import torch.multiprocessing as mp

from trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help="learning rate for the trainer")
    parser.add_argument('--batch_size', default=100, type=int, 
                        help="batch size for dataloader")
    parser.add_argument('--n_workers', default=0, type=int, 
                        help="number of workers to the dataloader")
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))