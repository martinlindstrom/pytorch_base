# Here, it is assumed that this ONLY runs on CPU, and that all necessary packages are available
# If this is intended to run via SLURM, see the example configuration in 'example_multi_gpu.sh'

# Run main script using the container
# To show all available options, run 'python3 main.py --help'
# Below are some options you might want to specify
# Notice especially the '--loader-workers', consider tuning this
# to fit your systems's capabilities
python3 main.py \
    --cpu-only \
    --dataset mnist \
    --data-path /path/to/dataset \
    --arch small_network \
    --batch-size 256 \
    --checkpoint /path/to/checkpoint/dir \
    --out-dim 10 \
    --epochs 200 \
    --checkpoint-freq 5 \
    --loader-workers LOADERS \ 
