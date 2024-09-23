# Torch-related imports
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler, DataLoader
import torch.distributed as dist

# NumPy etc.
import numpy as np
import matplotlib.pyplot as plt

# Standard library imports
from datetime import datetime
import argparse
import os

# Project imports
from models import SmallNetwork, ResNet18, ResNet34, ResNet50
from datasets import get_MNIST, get_CIFAR10, get_CIFAR100, get_imagenet

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
        self.inputs = torch.randn(size, 20)
        self.labels = torch.randint(10, size=(size,))

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

def parse_arguments():
    parser = argparse.ArgumentParser(description="=============================================\nPyTorch Training\n\nThis programme is compatible with both single-node multi-GPU, single-GPU, and CPU-only operation.\n\n\
For single-node multi-GPU operations, run this script with 'torchrun' AND use the '--multi-gpu' flag. See PyTorch docs for more information about torchrun.\n\n\
For single-GPU OR CPU-only operations, run with the usual 'python3' command.\n=============================================",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Dataset and loading
    dataloading = parser.add_argument_group(title="Data Handling", description="Dataset, path, and loading arguments")
    dataloading.add_argument('--data-path', metavar='DATAPATH', type=str,
                        help='path to dataset (default: not specified, which works with dummy dataset)')
    dataloading.add_argument('--dataset', metavar='DATASET', type=str, default="dummy",
                        help='name of dataset to load (default: dummy)')
    dataloading.add_argument('--loader-workers', default=10, type=int, metavar='LOADERS',
                        help='number of data loading workers (default: 10)')
    
    # Architecture and training/eval parameters
    arch_and_training = parser.add_argument_group(title="Architecture and Trainining/Eval", description="Architecture and training arguments")
    arch_and_training.add_argument('-a', '--arch', metavar='ARCHITECHTURE', default='dummy',
                        help='model architecture (default: dummy)')
    arch_and_training.add_argument('-d', '--out-dim', default=10, type=int, metavar='OUTDIM',
                        help='output dimension of the network (default: 10)')
    arch_and_training.add_argument('-e', '--epochs', default=200, type=int, metavar='EP',
                        help='number of total epochs to run (default: 200)')
    arch_and_training.add_argument('-b', '--batch-size', default=256, type=int, metavar='BATCHSIZE',
                        help='mini-batch size (default: 256), this is the total \
                            batch size across all GPUs when using multiple GPUs (default: 256)')
    arch_and_training.add_argument('-o', '--optimiser', default="sgd", type=str, metavar="OPTIM",
                        help="optimiser (default: sgd)")
    arch_and_training.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)', dest='lr')
    arch_and_training.add_argument('--momentum', default=0.9, type=float, metavar='MOM',
                        help='momentum (default: 0.9)')
    arch_and_training.add_argument('--weight-decay', default=5e-4, type=float,
                        metavar='WD', help='weight decay (default: 5e-4)')
    arch_and_training.add_argument('-s', '--scheduler', default="cosine_no_restart", type=str, metavar="SCHED",
                        help="learning rate scheduler (default: cosine_no_restart)")
    arch_and_training.add_argument('--warmup-epochs', default=0, type=int, metavar="WARMUP",
                        help="warmup epochs before decreasing learning according to scheduler (default: 0)")
    arch_and_training.add_argument('-l', '--loss-function', default="crossentropy", type=str, metavar="LOSS",
                        help="loss function (default: crossentropy)")
    arch_and_training.add_argument('-k', '--topk', default=[1,5], type=list, metavar="TOPK",
                        help="top-k accuracies to track (default: [1,5])")
    
    # Checkpointing and resuming
    checkpointing = parser.add_argument_group(title="Checkpointing and Resuming", description="Checkpointing and resuming paths and frequencies")
    checkpointing.add_argument('--resume-evaluate-model', default=False, type=str, metavar='MODELPATH',
                        help='path to model to load (default: False)')
    checkpointing.add_argument('-f', '--checkpoint-freq', metavar='FREQ', default=5, type=int,
                        help='eval and checkpoint every FREQ training epochs (default: 5)')
    checkpointing.add_argument('-c','--checkpoint', default="", type=str, metavar="CKPTPATH",
                        help="path to directory to store checkpoints; if empty string (default), don't checkpoint")
    
    
    # Multi-GPU/single-GPU/CPU training
    training_group = parser.add_argument_group(title="Training/Inference Style", description="Multi-GPU/Single-GPU/CPU. Pick exactly one.")
    mut_excl_training = training_group.add_mutually_exclusive_group(required=True)
    mut_excl_training.add_argument("--multi-gpu", action='store_true', 
                        help="Use all available GPUs on the current node. \
                            This flag assumes the file was run with 'torchrun'.")
    mut_excl_training.add_argument("--single-gpu", action='store_true',
                        help="Only use GPU 0 to train.")
    mut_excl_training.add_argument("--cpu-only", action='store_true',
                        help="Train only on CPU.")
    
    return parser.parse_args()

def set_up_savedir(args):
    # Only allow to re-use a savedir if resuming
    # Try to create the directory
    # If problematic, then only allow it if resuming from a checkpoint
    try: #attempting creation
        os.mkdir(args.checkpoint)
        print(f"Created checkpoint directory '{os.path.abspath(args.checkpoint)}'")
    except FileExistsError as e: #if dir exists
        if args.resume: #check that we are resuming a checkpoint in the checkpoint directory, to avoid accidental overwrite
            dir_path = os.path.abspath(args.checkpoint)
            ckpt_path = os.path.abspath(args.resume)
            if os.path.dirname(ckpt_path) == dir_path:
                print(f"Reusing checkpoint directory '{dir_path}'")
            else: #suppress FileExistError, below is the true cause
                raise RuntimeError(f"Cannot resume '{ckpt_path}' using the checkpoint directory '{dir_path}' since the desired checkpoint is not in the specified checkpoint directory.") from None
        else: #problem
            print(f"===Tried creating '{os.path.abspath(args.checkpoint)}' but failed. Does it already exist?===")
            raise e

def do_logging(logger, epoch, loss, accs, split, topk):
    # Print loss/accs in a nice format
    print(f"\t{split} Loss: {loss:.4e}", end=" ")
    for i in range(len(topk)):
        print(f"-- {split} Top-{topk[i]} Acc: {accs[i]*100:.4f}%", end=" ")
    print()
    # If logging is desired, then print the same values to the log
    if logger:
        logger.add_scalar(f"{split}/Loss", loss, epoch)
        for i in range(len(topk)):
            logger.add_scalar(f"{split}/Top{topk[i]}-Acc", accs[i], epoch)

def get_dataset(args, testset_only=False):
    if args.dataset == "dummy":
        return DummyDataset(size = 5000), DummyDataset(size = 1000), DummyDataset(size = 1000)
    elif args.dataset == "mnist":
        return get_MNIST(args, testset_only, valsplit=0.1)
    elif args.dataset == "cifar10":
        return get_CIFAR10(args, testset_only, valsplit=0.1)
    elif args.dataset == "cifar100":
        return get_CIFAR100(args, testset_only, valsplit=0.1)
    elif args.dataset == "imagenet":
        return get_imagenet(args, testset_only, valsplit=0.1)
    else:
        raise NotImplementedError(f"The dataset '{args.dataset}' is not implemented.")
    
def get_testloader_sampler(args):
    # Get ONLY testloader. Useful in, for example, evaluating trained models
    _, _, testset = get_dataset(args, testset_only=True)
    # Create sampler
    if args.multi_gpu: #need a distributed sampler. No need to shuffle validation/test sets
        testsampler = DistributedSampler(testset, shuffle=False, drop_last=False)
    else: #setting shuffle=true is the typical solution (and is equivalent), but to make the code more modular, instantiate a random sampler manually
        testsampler = SequentialSampler(testset)
        
    # Now create the loaders
    # There are general data loading optimisations which are done, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html 
    if args.multi_gpu:
        # Correctly scale batch size per loader
        # Also use pin_memory for better performance
        testloader = DataLoader(testset, sampler=testsampler, batch_size=int(args.batch_size/dist.get_world_size()), num_workers=args.loader_workers, pin_memory=True)
    elif args.single_gpu:
        # Pin memory for better performance
        testloader = DataLoader(testset, sampler=testsampler, batch_size=args.batch_size, num_workers=args.loader_workers, pin_memory=True)
    else: #Then args.cpu_only:
        testloader = DataLoader(testset, sampler=testsampler, batch_size=args.batch_size, num_workers=args.loader_workers)
    return testloader, testsampler

def get_dataloaders_samplers(args):
    # Get train, val, and test sets
    trainset, valset, testset = get_dataset(args)
    # Create samplers
    if args.multi_gpu: #need a distributed sampler. No need to shuffle validation/test sets
        trainsampler = DistributedSampler(trainset, shuffle=True, drop_last=False)
        valsampler = DistributedSampler(valset, shuffle=False, drop_last=False)
        testsampler = DistributedSampler(testset, shuffle=False, drop_last=False)
    else: #setting shuffle=true is the typical solution (and is equivalent), but to make the code more modular, instantiate a random sampler manually
        trainsampler = RandomSampler(trainset)
        valsampler = SequentialSampler(valset)
        testsampler = SequentialSampler(testset)
        
    # Now create the loaders
    # There are general data loading optimisations which are done, see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html 
    if args.multi_gpu:
        # Correctly scale batch size per loader
        # Also use pin_memory for better performance
        trainloader = DataLoader(trainset, sampler=trainsampler, batch_size=int(args.batch_size/dist.get_world_size()), num_workers=args.loader_workers, pin_memory=True)
        valloader = DataLoader(valset, sampler=valsampler, batch_size=int(args.batch_size/dist.get_world_size()), num_workers=args.loader_workers, pin_memory=True)
        testloader = DataLoader(testset, sampler=testsampler, batch_size=int(args.batch_size/dist.get_world_size()), num_workers=args.loader_workers, pin_memory=True)
    elif args.single_gpu:
        # Pin memory for better performance
        trainloader = DataLoader(trainset, sampler=trainsampler, batch_size=args.batch_size, num_workers=args.loader_workers, pin_memory=True)
        valloader = DataLoader(valset, sampler=valsampler, batch_size=args.batch_size, num_workers=args.loader_workers, pin_memory=True)
        testloader = DataLoader(testset, sampler=testsampler, batch_size=args.batch_size, num_workers=args.loader_workers, pin_memory=True)
    else: #Then args.cpu_only:
        trainloader = DataLoader(trainset, sampler=trainsampler, batch_size=args.batch_size, num_workers=args.loader_workers)
        valloader = DataLoader(valset, sampler=valsampler, batch_size=args.batch_size, num_workers=args.loader_workers)
        testloader = DataLoader(testset, sampler=testsampler, batch_size=args.batch_size, num_workers=args.loader_workers)
    return trainloader, trainsampler, valloader, valsampler, testloader, testsampler

def get_model(args):
    if args.arch == "dummy":
        return nn.Linear(20,args.out_dim)
    elif args.arch == "small_network":
        return SmallNetwork(out_dim=args.out_dim)
    elif args.arch == "resnet18":
        return ResNet18(out_dim=args.out_dim)
    elif args.arch == "resnet34":
        return ResNet34(out_dim=args.out_dim)
    elif args.arch == "resnet50":
        return ResNet50(out_dim=args.out_dim)
    else:
        raise NotImplementedError(f"The model '{args.arch}' is not implemented.")

def get_optimiser(model, args):
    if args.optimiser == "sgd":
        return SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"The optimiser '{args.optimiser}' is not implemented.")
    
def get_scheduler(optimiser, args):
    # Get main scheduler
    if args.scheduler == "cosine_no_restart":
        main_scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs-args.warmup_epochs)
    else:
        raise NotImplementedError(f"The scheduler '{args.scheduler}' is not implemented.")
    # Optionally add warmup
    if args.warmup_epochs > 0:
        warmup_scheduler = ConstantLR(optimiser, factor=1) #constant at starting lr
        scheduler = SequentialLR(optimiser, [warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])
        return scheduler
    else:
        return main_scheduler

def get_loss(args):
    if args.loss_function == "crossentropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function {args.loss_function} is not implemented.")

def save_model(model, args, filename="final_model.pth"):
    final_filename = os.path.join(args.checkpoint, filename)
    torch.save({
        "parameters" : model.module.state_dict() if args.multi_gpu else model.state_dict() #If multi-GPU, undo DDP wrapper
    }, final_filename)

def load_model(model, model_path, device):
    if os.path.exists(model_path):
        print(f"Loading checkpoint from '{os.path.abspath(model_path)}' ...", end=" ")
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict["parameters"])
        print("done!")
        return model
    else:
        raise RuntimeError(f"The model at '{os.path.abspath(model_path)}' could not be found.")

def save_checkpoint(epoch, model, optimiser, scheduler, args, filename = "checkpoint.pth"):
    checkpoint_filename = os.path.join(args.checkpoint, filename)
    torch.save({
        "epoch" : epoch,
        "parameters" : model.module.state_dict() if args.multi_gpu else model.state_dict(), #If multi-GPU, undo DDP wrapper
        "optimiser" : optimiser.state_dict(),
        "scheduler" : scheduler.state_dict()
    }, checkpoint_filename)

def load_checkpoint(model, optimiser, scheduler, checkpoint_path, device):
    # Check if the save file exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from '{os.path.abspath(checkpoint_path)}' ...", end=" ")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["parameters"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]+1 #continue from next epoch 
        print("done!")
        return model, optimiser, scheduler, start_epoch
    else:
        raise RuntimeError(f"The checkpoint at '{os.path.abspath(checkpoint_path)}' could not be found.")

def run_epoch(epoch, dataloader, sampler, model, optimiser, scheduler, loss_fcn, device, args):
    # Setup
    if args.multi_gpu:
        sampler.set_epoch(epoch) #For correct shuffling in distributed mode
    model.train()

    # Loop over batches
    for batch, (x,y) in enumerate(dataloader):
        # Load data onto device
        x = x.to(device)
        y = y.to(device)
        # Training
        optimiser.zero_grad(set_to_none=True) 
        out = model(x)
        loss = loss_fcn(out, y)
        loss.backward()
        optimiser.step()
    
    # Post-action
    scheduler.step()

def batch_topk(out, y, topk, device):
    # Returns the top-k correct for all specified values in 'topk'
    with torch.no_grad():
        maxk = torch.max(torch.tensor(topk, device=device))
        batch_size = out.shape[0]

        _, kpred = out.topk(k=maxk, dim=1, largest=True, sorted=True) #(batch_size x maxk) shape
        correct = kpred.eq(y.view(-1, 1).expand_as(kpred)) #(batch_size x maxk) shape

        res = torch.zeros_like(torch.tensor(topk), device=device)
        for i in range(len(topk)):
            k = topk[i]
            correct_k = correct[:,:k].view(-1).sum(0) # (batch_size) shape
            res[i] = correct_k.sum()
        return res

def evaluate(epoch, dataloader, sampler, model, loss_fcn, device, args):
    print(f"{device} - dataset len:{len(dataloader.dataset)}")
    print(f"{device} - batch size:{dataloader.batch_size}")
    print(f"{device} - sampler len:{len(sampler)}")
    # Setup
    if args.multi_gpu:
        sampler.set_epoch(epoch)
    model.eval()
    loss = torch.tensor(0., device=device)
    topk_correct = torch.zeros_like(torch.tensor(args.topk), device=device) #(args.topk,) shape
    
    # Loop over batches
    with torch.no_grad():
        for batch, (x,y) in enumerate(dataloader):
            # Load data onto device
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            # Track loss
            loss += loss_fcn(out, y)
            # Decode
            temp_correct = batch_topk(out, y, args.topk, device)
            print(f"{device}:{batch} - accs:{temp_correct}")
            topk_correct += temp_correct
    print("\n============After loop===========\n\n")
    print(f"{device} - {topk_correct}")
    if args.multi_gpu:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(topk_correct, op=dist.ReduceOp.SUM)
    print("\n============After reduce===========\n\n")
    print(f"{device} - {topk_correct}")
        
    # Post-action: normalise correctly
    loss = loss/len(dataloader.dataset)
    topk_accuracy = topk_correct/len(dataloader.dataset)

    print("\n============After normalisation===========\n\n")
    print(f"{device} - {topk_accuracy}")

    return loss, topk_accuracy


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    bs = 256
    dim = 100
    y = torch.randint(high=10, size=(bs,))
    temp = torch.randn((256,100))
    out = nn.functional.softmax(temp, dim=1)

        
