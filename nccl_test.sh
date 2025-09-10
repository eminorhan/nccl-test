#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=nccl_test
#SBATCH --output=nccl_test_%A_%a.out
#SBATCH --array=0

# activate venv
source /lustre/gale/stf218/scratch/emin/ncclvenv/bin/activate

# set misc env vars
export LD_LIBRARY_PATH=/lustre/gale/stf218/scratch/emin/aws-ofi-nccl-1.14.0/lib:$LD_LIBRARY_PATH  # enable aws-ofi-nccl
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/lustre/gale/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/gale/stf218/scratch/emin/pytorch_kernel_cache"
export GPUS_PER_NODE=4

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node 4 --max_restarts 9 --node_rank $SLURM_NODEID --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" ./nccl_test.py

echo "Done"