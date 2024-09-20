#! /bin/env bash
# Example SLURM sbatch configurations one could set
#SBATCH -A PROJID                                               # Project ID, might be mandatory
#SBATCH -t D-HHMMSS                                             # Max runtime
#SBATCH --gpus-per-node=TYPE:NUMBER				# GPU choice
#SBATCH -J IDENTIFIER                                           # Run identifier
#SBATCH -o /PATH/TO/FILE                                        # Output file for stdout
#SBATCH --mail-type=END,FAIL                                    # Get email notifications for events, here END or FAIL of job
#SBATCH --mail-user=your.email@somedomain.tld                   # Email address to send notifications to

# Use a Singularity container
# Purge other modules if such exist (remove if not available on your cluster)
module purge
CONTAINER=/path/to/container.sif

# Dataset paths
imagenet_path=/path/to/imagenet

# Environment variables setup
# Export number of GPUs to programme
ngpus=$SLURM_GPUS_ON_NODE
export WORLD_SIZE=$ngpus
# It is useful to set OMP_NUM_THREADS to speed up data loading
# Modify this according to your system's capabilities
cores_per_gpu=16
num_cores=$(($ngpus*$cores_per_gpu))
export OMP_NUM_THREADS=$num_cores

# Run main script using the container
# To show all available options, run 'python3 main.py --help'
# Below are some options you might want to specify
# Notice especially the '--loader-workers', consider tuning this
# to fit your systems's capabilities
singularity exec $CONTAINER torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$ngpus \
    main.py \
    --multi-gpu \
    --dataset imagenet \
    --data-path $imagenet_path \
    --arch resnet50 \
    --batch-size 256 \
    --checkpoint /path/to/checkpoint/dir \
    --out-dim 1000 \
    --epochs 200 \
    --checkpoint-freq 5 \
    --loader-workers $cores_per_gpu \ 
