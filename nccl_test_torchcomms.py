import os
import torch
import time
import argparse
from torchcomms import new_comm, ReduceOp

def setup_distributed(backend):
    """
    Initializes the distributed environment using torchcomms.
    """
    # These environment variables are set by torchrun
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    print(f"Rank/world_size/local_rank: {rank}/{world_size}/{local_rank}")

    torch.cuda.set_device(local_rank)
    target_device = torch.device(f"cuda:{local_rank}")
    torchcomm = new_comm(backend, target_device, name="bench_comm")
    
    return torchcomm, rank, world_size, local_rank


def benchmark_all_reduce(torchcomm, tensor_size_mb, num_warmup, num_iters, rank, world_size):
    """
    Benchmarks the all_reduce operation using torchcomms on CUDA.
    """
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4  # Assuming float32 (4 bytes per element)
    
    # Create a tensor on the correct device
    data = torch.randn(num_elements, dtype=torch.float32, device=torch.cuda.current_device())

    # Ensure all processes are ready before starting the benchmark
    torchcomm.barrier(async_op=False)

    # Warm-up iterations to exclude initialization overhead from timing
    if rank == 0:
        print(f"🚀 Starting {num_warmup} warm-up iterations...")
    for _ in range(num_warmup):
        torchcomm.all_reduce(data, ReduceOp.AVG, async_op=False)

    # Wait for all warm-up device operations to complete
    torch.cuda.current_stream().synchronize()

    # Timed iterations
    if rank == 0:
        print(f"🔥 Starting benchmark with tensor size: {tensor_size_mb} MB...")
    
    # Sync all processes before starting the timer
    torchcomm.barrier(async_op=False)
    start_time = time.perf_counter()

    for _ in range(num_iters):
        torchcomm.all_reduce(data, ReduceOp.AVG, async_op=False)
    
    # Wait for all device operations to finish before stopping the timer
    torch.cuda.current_stream().synchronize()
    end_time = time.perf_counter()
    
    # Final sync before calculating and printing results
    torchcomm.barrier(async_op=False)

    # Calculate results
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iters
    
    # Bus bandwidth = (tensor Size in bits) / time in seconds
    bus_bandwidth_gbps = (tensor_size_bytes * 8) / (avg_time_per_iter * 1e9)
    
    if rank == 0:
        print("\n--- 📊 Results ---")
        print(f"World size (total GPUs): {world_size}")
        print(f"Tensor size: {tensor_size_mb} MB")
        print(f"Average time per all_reduce: {avg_time_per_iter * 1000:.4f} ms")
        print(f"Bus bandwidth: {bus_bandwidth_gbps:.4f} Gbps")
        print("--------------------")


def main():
    parser = argparse.ArgumentParser(description="torchcomms all_reduce benchmark")
    parser.add_argument('--size', type=int, default=90000, help='Tensor size in MB to be reduced.')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warm-up iterations.')
    parser.add_argument('--iters', type=int, default=5, help='Number of timed iterations.')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'ncclx'], help='Torchcomms communication backend (CUDA only).')
    
    args = parser.parse_args()

    torchcomm, rank, world_size, local_rank = setup_distributed(backend=args.backend)
    
    try:
        benchmark_all_reduce(
            torchcomm=torchcomm,
            tensor_size_mb=args.size, 
            num_warmup=args.warmup,
            num_iters=args.iters,
            rank=rank,
            world_size=world_size
        )
    finally:
        # Guarantee cleanup runs even if the benchmark fails
        torchcomm.finalize()

if __name__ == "__main__":
    main()