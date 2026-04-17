#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --job-name=test_torchcomms
#SBATCH --output=test_torchcomms_%A_%a.out
#SBATCH --array=0

# activate venv
source /lustre/blizzard/stf218/scratch/emin/blizzardvenv/bin/activate

# set misc env vars
export LD_LIBRARY_PATH=/lustre/blizzard/stf218/scratch/emin/aws-ofi-nccl-1.19.0/lib:$LD_LIBRARY_PATH  # enable aws-ofi-nccl
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/lustre/blizzard/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/blizzard/stf218/scratch/emin/pytorch_kernel_cache"
export GPUS_PER_NODE=4

export NCCL_COLLTRACE=0
export NCCL_CTRAN_BACKENDS="NVL,SOCKET"

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

srun python ./test_torchcomms.py

echo "Done"