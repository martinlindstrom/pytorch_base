# Here, it is assumed that this doesn't run on a multi-GPU system, and that all necessary packages are available
# In particular, the code defaults to use GPU 0 with the '--single-gpu' option
# If this does not suit you, consider changing this, or modifying CUDA_VISIBLE_DEVICES
# If this is intended to run via SLURM, see the example configuration in 'example_multi_gpu.sh'

# Run main script using the container
# To show all available options, run 'python3 main.py --help'
# Below are some options you might want to specify
# Notice especially the '--loader-workers', consider tuning this
# to fit your systems's capabilities
python3 main.py \
    --multi-gpu \
    --dataset cifar10 \
    --data-path /path/to/dataset \
    --arch resnet18 \
    --batch-size 256 \
    --checkpoint /path/to/checkpoint/dir \
    --out-dim 10 \
    --epochs 200 \
    --checkpoint-freq 5 \
    --loader-workers LOADERS \ 
