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
from utils import (parse_arguments, set_up_savedir, do_logging,
    get_model, get_optimiser, get_scheduler, get_loss, get_dataloaders_samplers,
    save_checkpoint, load_checkpoint, save_model,
    run_epoch, evaluate)

"""
TODO:
- Investigate TF32 instead of FP32
- Investigate A100/A40 epoch stats for different number of GPUs
"""

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

    # Set up model
    model = get_model(args)
    # DDP-ify model if needed
    if args.multi_gpu:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model.to(device)

    # Set up optimiser, scheduler, dataset, loss
    optimiser = get_optimiser(model, args)
    scheduler = get_scheduler(optimiser, args)
    trainloader, trainsampler, valloader, valsampler, testloader, testsampler = get_dataloaders_samplers(args)
    loss_fcn = get_loss(args)

    # Resume from checkpoint if this is desired
    if args.resume: #load checkpoint which was aborted from
        model, optimiser, scheduler, start_epoch = load_checkpoint(model, optimiser, scheduler, args.resume, device)
    else: #default values: anything non-randomised
        start_epoch = 0

    # Set up checkpointing and logging if desired
    # Throws an exception if the specified checkpoint dir already exists; 
    # this avoids overwriting past logs/checkpoints
    # Only do logging if checkpoint dir is specified
    # Evals are always printed no matter what
    if args.checkpoint:
        if args.multi_gpu:
            if rank==0:
                set_up_savedir(args)
                logger = SummaryWriter(args.checkpoint, filename_suffix=".log")
        else:
            set_up_savedir(args)
            logger = SummaryWriter(args.checkpoint, filename_suffix=".log")
    else:
        logger = None

    # Main train loop
    if args.multi_gpu:
        if rank == 0:
            print(f"=============================================\nStarting training\n=============================================")
    else:
        print(f"=============================================\nStarting training\n=============================================")
    for epoch in range(start_epoch, args.epochs):
        # Run epoch with informative prints
        if args.multi_gpu:
            if rank == 0:
                print(f"Epoch {epoch} / {args.epochs-1} -- ", end="")
                epoch_start_time = datetime.now()
        else:
            print(f"Epoch {epoch} / {args.epochs-1} -- ", end="")
            epoch_start_time = datetime.now()
        run_epoch(epoch, trainloader, trainsampler, model, optimiser, scheduler, loss_fcn, device, args)
        if args.multi_gpu:
            if rank == 0:
                epoch_time = datetime.now()-epoch_start_time
                print(str(epoch_time).split(".")[0], "s")
                if logger:
                    logger.add_scalar("Time/Train", epoch_time.total_seconds(), epoch)
        else:
            epoch_time = datetime.now()-epoch_start_time
            print(str(epoch_time).split(".")[0], "s")
            if logger:
                logger.add_scalar("Time/Train", epoch_time.total_seconds(), epoch)

        # Eval and logging every args.checkpoint_freq with informative prints
        if epoch % args.checkpoint_freq == 0:
            if args.multi_gpu:
                if rank == 0:
                    print(f"=== Eval at epoch {epoch} ===")
                    eval_start_time = datetime.now()
            else:
                print(f"=== Eval at epoch {epoch} ===")
                eval_start_time = datetime.now()
            
            # Evaluate and log metrics
            train_loss, train_acc = evaluate(epoch, trainloader, trainsampler, model, loss_fcn, device, args)
            val_loss, val_acc = evaluate(epoch, valloader, valsampler, model, loss_fcn, device, args)
            if args.multi_gpu:
                if rank==0:
                    do_logging(logger, epoch, train_loss, train_acc, "Train", args.topk)
                    do_logging(logger, epoch, val_loss, val_acc, "Val", args.topk)
            else:
                do_logging(logger, epoch, train_loss, train_acc, "Train", args.topk)
                do_logging(logger, epoch, val_loss, val_acc, "Val", args.topk)
            
            # Checkpoint if desired
            if args.checkpoint:
                if args.multi_gpu:
                    if rank==0:
                        save_checkpoint(epoch, model, optimiser, scheduler, args)
                else:
                    save_checkpoint(epoch, model, optimiser, scheduler, args)

            if args.multi_gpu:
                if rank == 0:
                    eval_time = datetime.now() - eval_start_time
                    print(f"=== Eval duration: {str(eval_time).split('.')[0]} s ===")
                    if logger:
                        logger.add_scalar("Time/Eval", eval_time.total_seconds(), epoch)
            else:
                eval_time = datetime.now() - eval_start_time
                print(f"=== Eval duration: {str(eval_time).split('.')[0]} s ===")
                if logger:
                    logger.add_scalar("Time/Eval", eval_time.total_seconds(), epoch)
            
    # After training: Final eval and model save
    if args.multi_gpu:
        if rank == 0:
            print(f"=============================================\nTesting\n=============================================")
    else:
        print(f"=============================================\nTesting\n=============================================")
    test_loss, test_acc = evaluate(args.epochs, testloader, testsampler, model, loss_fcn, device, args)
    if args.multi_gpu:
        if rank==0:
            do_logging(logger, args.epochs, test_loss, test_acc, "Test", args.topk)
    else:
        do_logging(logger, args.epochs, test_loss, test_acc, "Test", args.topk)
    
    # Save if desired
    if args.checkpoint:
        if args.multi_gpu:
            if rank==0:
                save_model(model, args)
        else:
            save_model(model, args)

    # Close the logger
    if args.multi_gpu:
        if rank==0:
            if logger:
                logger.close()
    else:
        if logger:
            logger.close()

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
