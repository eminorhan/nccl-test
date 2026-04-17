import os
import torch
import time
import argparse
from torchcomms import new_comm, ReduceOp

def setup_distributed(backend):
    """
    Initializes the distributed environment using torchcomms via Slurm variables.
    """
    # These environment variables are standard for Slurm deployments
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    
    if rank == 0:
        print(f"Rank/world_size/local_rank: {rank}/{world_size}/{local_rank}")

    torch.cuda.set_device(local_rank)
    target_device = torch.device(f"cuda:{local_rank}")
    
    # Initialize torchcomms
    torchcomm = new_comm(backend, target_device, name="bench_comm")
    
    return torchcomm, rank, world_size, local_rank


def benchmark_primitive(torchcomm, primitive, tensor_size_mb, dtype_str, num_warmup, num_iters, rank, world_size):
    """
    Benchmarks the specified communication primitive using torchcomms.
    """
    device = torch.cuda.current_device()
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    
    # Map string to PyTorch dtype and determine bytes per element
    dtype_map = {
        'float32': (torch.float32, 4),
        'float16': (torch.float16, 2),
        'bfloat16': (torch.bfloat16, 2)
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
        
    torch_dtype, bytes_per_element = dtype_map[dtype_str]
    
    # Calculate elements to match requested MB size exactly, divisible by world_size
    num_elements = (tensor_size_bytes // bytes_per_element) // world_size * world_size 
    actual_tensor_size_bytes = num_elements * bytes_per_element
    
    # Allocate tensors based on the primitive and chosen dtype
    if primitive in ['all_reduce', 'broadcast', 'reduce']:
        data = torch.randn(num_elements, dtype=torch_dtype, device=device)
        tensor_list = None
    elif primitive == 'all_gather':
        chunk_elements = num_elements // world_size
        data = torch.randn(chunk_elements, dtype=torch_dtype, device=device)
        tensor_list = [torch.empty(chunk_elements, dtype=torch_dtype, device=device) for _ in range(world_size)]
    elif primitive == 'reduce_scatter':
        chunk_elements = num_elements // world_size
        data = torch.empty(chunk_elements, dtype=torch_dtype, device=device)
        tensor_list = [torch.randn(chunk_elements, dtype=torch_dtype, device=device) for _ in range(world_size)]
    else:
        raise ValueError(f"Unknown primitive: {primitive}")

    def execute_op():
        """Wrapper to handle the varying signatures of torchcomms primitives."""
        if primitive == 'all_reduce':
            torchcomm.all_reduce(data, ReduceOp.AVG, async_op=False)
        elif primitive == 'all_gather':
            torchcomm.all_gather(tensor_list, data, async_op=False)
        elif primitive == 'reduce_scatter':
            torchcomm.reduce_scatter(data, tensor_list, ReduceOp.AVG, async_op=False)
        elif primitive == 'broadcast':
            torchcomm.broadcast(data, src=0, async_op=False)
        elif primitive == 'reduce':
            torchcomm.reduce(data, dst=0, op=ReduceOp.AVG, async_op=False)

    # Ensure all processes are ready before starting
    torchcomm.barrier(async_op=False)

    if rank == 0:
        print(f"🚀 Starting {num_warmup} warm-up iterations for {primitive} ({dtype_str})...")
    
    # Warm-up phase
    for _ in range(num_warmup):
        execute_op()

    torch.cuda.current_stream().synchronize()

    if rank == 0:
        print(f"🔥 Starting benchmark with total tensor size: ~{tensor_size_mb} MB...")
    
    # Timed phase
    torchcomm.barrier(async_op=False)
    start_time = time.perf_counter()

    for _ in range(num_iters):
        execute_op()
    
    torch.cuda.current_stream().synchronize()
    end_time = time.perf_counter()
    torchcomm.barrier(async_op=False)

    # Calculate results
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iters
    
    # Algorithmic bandwidth calculation (Gbps)
    alg_bw_gbps = (actual_tensor_size_bytes * 8) / (avg_time_per_iter * 1e9)
    
    # Bus bandwidth correction factors based on the algorithm topology
    if primitive == 'all_reduce':
        bus_correction = 2 * (world_size - 1) / world_size
    elif primitive in ['all_gather', 'reduce_scatter']:
        bus_correction = (world_size - 1) / world_size
    else:
        bus_correction = 1.0 # Broadcast and reduce don't use ring topologies the same way
        
    bus_bandwidth_gbps = alg_bw_gbps * bus_correction
    
    if rank == 0:
        print("\n--- 📊 Results ---")
        print(f"Primitive: {primitive}")
        print(f"Data Type: {dtype_str}")
        print(f"Backend: {torchcomm.get_backend()}")
        print(f"World size: {world_size} GPUs")
        print(f"Total payload size: {actual_tensor_size_bytes / (1024*1024):.2f} MB")
        print(f"Average time per iter: {avg_time_per_iter * 1000:.4f} ms")
        print(f"Algorithmic bandwidth: {alg_bw_gbps:.4f} Gbps")
        print(f"Calculated Bus bandwidth: {bus_bandwidth_gbps:.4f} Gbps")
        print("--------------------")


def main():
    parser = argparse.ArgumentParser(description="torchcomms communication benchmark")
    parser.add_argument('--primitive', type=str, default='all_reduce', choices=['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'reduce'], help='The communication primitive to benchmark.')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'], help='Data type of the tensors.')
    parser.add_argument('--size', type=int, default=10000, help='Total tensor size in MB across the operation.')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warm-up iterations.')
    parser.add_argument('--iters', type=int, default=5, help='Number of timed iterations.')
    parser.add_argument('--backend', type=str, default='ncclx', choices=['nccl', 'ncclx'], help='Torchcomms communication backend (CUDA only).')
    
    args = parser.parse_args()

    torchcomm, rank, world_size, local_rank = setup_distributed(backend=args.backend)
    
    try:
        benchmark_primitive(
            torchcomm=torchcomm,
            primitive=args.primitive,
            tensor_size_mb=args.size, 
            dtype_str=args.dtype,
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