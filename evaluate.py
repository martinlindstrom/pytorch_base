# Torch-related imports
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

# NumPy etc.
import numpy as np
import matplotlib.pyplot as plt

# Standard library imports
from datetime import datetime
import os

# Project imports
from utils import (parse_arguments, do_logging,
    get_model, get_optimiser, get_scheduler, get_loss, get_testloader_sampler,
    load_model, 
    evaluate)

def setup(backend="nccl", verbose=True):
    """Setup the multi-GPU environment to be compatible with torchrun"""
    try:
        dist.init_process_group(backend)
    except ValueError as e:
        print(f"!!!!!\nCould not initialise process group. Expected the file to be run with 'torchrun', has this been done?\n!!!!!\n", e)
    if verbose:
        print(f'''
=============================================
Rank:          {dist.get_rank()}
World size:    {dist.get_world_size()}
Master addres: {os.environ["MASTER_ADDR"]}
Master port:   {os.environ["MASTER_PORT"]}
=============================================
        ''')

def cleanup():
    """Necessary clean-up to end torchrun run."""
    dist.destroy_process_group()

def main(args):
    """Main: This either runs on multiple GPU:s, or on a single GPU, or on CPU.
    
    Firstly, if multi-GPU is desired, check whether the 'torchrun' settings work. 
    Then, set up correct device (for all run settings). This gives a run which is (mostly) the same no matter run setting.

    Models, data, and logging require minor tweaks to work correctly.
    """

    # Determine device
    if args.multi_gpu: #Handle multi-GPU device setup
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else: #Handle regular device setup
        device = torch.device("cuda:0") if args.single_gpu else torch.device("cpu")

    # Set up model, optimiser, scheduler, dataset, loss
    model = get_model(args)
    optimiser = get_optimiser(model, args)
    scheduler = get_scheduler(optimiser, args)
    testloader, testsampler = get_testloader_sampler(args)
    loss_fcn = get_loss(args)

    # Load model to evaluate
    if args.resume_evaluate_model: #load checkpoint which was aborted from
        model = load_model(model, args.resume_evaluate_model, device)
    else: # Throw error: this is the only purpose of this script
        raise ValueError(f"Did not get model to evaluate.")
    
    # DDP-ify model if needed
    if args.multi_gpu:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model.to(device)

    # Evaluate
    if args.multi_gpu:
        if rank == 0:
            print(f"=============================================\nEvaluating\n=============================================")
    else:
        print(f"=============================================\nEvaluating\n=============================================")
    test_loss, test_acc = evaluate(args.epochs, testloader, testsampler, model, loss_fcn, device, args)
    # Print
    logger = None
    if args.multi_gpu:
        if rank==0:
            do_logging(logger, args.epochs, test_loss, test_acc, "Test", args.topk)
    else:
        do_logging(logger, args.epochs, test_loss, test_acc, "Test", args.topk)

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # See if run options are ok
    if args.multi_gpu and not (torch.cuda.device_count() > 1): #multi-gpu needs more than 1 GPU
        raise RuntimeError(f"Wanted multi-GPU but only {torch.cuda.device_count()} GPUs are available. Consider using the '--single-gpu' or '--cpu-only' flags.")
    elif args.single_gpu and not (torch.cuda.device_count() >= 1): #single-gpu requires at least 1 GPU
        raise RuntimeError(f"Wanted single-GPU but could not find any GPU. Consider using the '--cpu-only' flag.")
    # CPU is always ok

    # Now we can setup, launch, and exit gracefully
    if args.multi_gpu:
        setup()
        record(main(args))
        cleanup()
    else:
        main(args)
