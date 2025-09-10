import torch
import torch.distributed as dist
import time
import os
import argparse

def setup_distributed():
    """
    Initializes the distributed environment using environment variables
    set by torchrun.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training is not available.")

    # These environment variables are set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the process group with the 'nccl' backend
    dist.init_process_group("nccl", init_method="env://")

    # Pin the process to the correct GPU
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()


def benchmark_all_reduce(tensor_size_mb, num_warmup, num_iters, rank, world_size):
    """
    Benchmarks the all_reduce operation.

    Args:
        tensor_size_mb (int): The size of the tensor in megabytes.
        num_warmup (int): Number of warm-up iterations.
        num_iters (int): Number of timed iterations.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    # Create a tensor on the correct GPU device
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4  # Assuming float32 (4 bytes per element)
    data = torch.randn(num_elements, dtype=torch.float32, device=torch.cuda.current_device())

    # Ensure all processes are ready before starting the benchmark
    dist.barrier()

    # Warm-up iterations to exclude initialization overhead from timing
    if rank == 0:
        print(f"🚀 Starting {num_warmup} warm-up iterations...")
    for _ in range(num_warmup):
        dist.all_reduce(data, op=dist.ReduceOp.AVG)

    # Wait for all warm-up GPU operations to complete across all devices
    torch.cuda.synchronize()

    # Timed iterations
    if rank == 0:
        print(f"🔥 Starting benchmark with tensor size: {tensor_size_mb} MB...")
    
    # Sync all processes before starting the timer
    dist.barrier()
    start_time = time.perf_counter()

    for _ in range(num_iters):
        dist.all_reduce(data, op=dist.ReduceOp.AVG)
    
    # Wait for all GPU operations to finish before stopping the timer
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    dist.barrier()

    # Calculate results
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iters
    
    # Bus bandwidth = (tensor Size in bits) / time in seconds
    # We convert MB to bits: tensor_size_mb * 1024 * 1024 * 8
    bus_bandwidth_gbps = (tensor_size_bytes * 8) / (avg_time_per_iter * 1e9)
    
    if rank == 0:
        print("\n--- 📊 Results ---")
        print(f"World size (total GPUs): {world_size}")
        print(f"Tensor size: {tensor_size_mb} MB")
        print(f"Average time per all_reduce: {avg_time_per_iter * 1000:.4f} ms")
        print(f"Bus bandwidth: {bus_bandwidth_gbps:.4f} Gbps")
        print("--------------------")


def main():
    parser = argparse.ArgumentParser(description="nccl dist.all_reduce() benchmark")
    parser.add_argument('--size', type=int, default=90000, help='Tensor size in MB to be reduced.')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warm-up iterations.')
    parser.add_argument('--iters', type=int, default=5, help='Number of timed iterations.')
    args = parser.parse_args()

    rank, world_size, _ = setup_distributed()
    benchmark_all_reduce(
        tensor_size_mb=args.size, 
        num_warmup=args.warmup,
        num_iters=args.iters,
        rank=rank,
        world_size=world_size
    )
    cleanup()

if __name__ == "__main__":
    main()