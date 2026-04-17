import torch
import torch.distributed as dist
import time
import os
import argparse

def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("Distributed training is not available.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def benchmark_primitive(primitive, tensor_size_mb, dtype_str, num_warmup, num_iters, rank, world_size):
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
        if primitive == 'all_reduce':
            dist.all_reduce(data, op=dist.ReduceOp.AVG)
        elif primitive == 'all_gather':
            dist.all_gather(tensor_list, data)
        elif primitive == 'reduce_scatter':
            dist.reduce_scatter(data, tensor_list, op=dist.ReduceOp.AVG)
        elif primitive == 'broadcast':
            dist.broadcast(data, src=0)
        elif primitive == 'reduce':
            dist.reduce(data, dst=0, op=dist.ReduceOp.AVG)

    # Ensure all processes are ready before starting
    dist.barrier()

    if rank == 0:
        print(f"🚀 Starting {num_warmup} warm-up iterations for {primitive} ({dtype_str})...")
    for _ in range(num_warmup):
        execute_op()

    torch.cuda.synchronize()

    if rank == 0:
        print(f"🔥 Starting benchmark with total tensor size: ~{tensor_size_mb} MB...")
    
    dist.barrier()
    start_time = time.perf_counter()

    for _ in range(num_iters):
        execute_op()
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    dist.barrier()

    # Calculate results
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iters
    
    alg_bw_gbps = (actual_tensor_size_bytes * 8) / (avg_time_per_iter * 1e9)
    
    if primitive == 'all_reduce':
        bus_correction = 2 * (world_size - 1) / world_size
    elif primitive in ['all_gather', 'reduce_scatter']:
        bus_correction = (world_size - 1) / world_size
    else:
        bus_correction = 1.0
        
    bus_bandwidth_gbps = alg_bw_gbps * bus_correction
    
    if rank == 0:
        print("\n--- 📊 Results ---")
        print(f"Primitive: {primitive}")
        print(f"Data Type: {dtype_str}")
        print(f"World size: {world_size} GPUs")
        print(f"Total payload size: {actual_tensor_size_bytes / (1024*1024):.2f} MB")
        print(f"Average time per iter: {avg_time_per_iter * 1000:.4f} ms")
        print(f"Algorithmic bandwidth: {alg_bw_gbps:.4f} Gbps")
        print(f"Calculated Bus bandwidth: {bus_bandwidth_gbps:.4f} Gbps")
        print("--------------------")


def main():
    parser = argparse.ArgumentParser(description="NCCL communication benchmark")
    parser.add_argument('--primitive', type=str, default='all_reduce', choices=['all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'reduce'])
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'], help='Data type of the tensors.')
    parser.add_argument('--size', type=int, default=10000, help='Total tensor size in MB.')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warm-up iterations.')
    parser.add_argument('--iters', type=int, default=5, help='Number of timed iterations.')
    args = parser.parse_args()

    rank, world_size, _ = setup_distributed()
    
    benchmark_primitive(
        primitive=args.primitive,
        tensor_size_mb=args.size, 
        dtype_str=args.dtype,
        num_warmup=args.warmup,
        num_iters=args.iters,
        rank=rank,
        world_size=world_size
    )
    
    cleanup()

if __name__ == "__main__":
    main()