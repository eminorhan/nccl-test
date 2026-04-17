### `nccl`/`ncclx` benchmark

A quick test to benchmark your `pytorch` + `nccl` setup for distributed training. Use the SLURM batch script [nccl_test.sh](nccl_test.sh) to run this benchmark on a SLURM cluster (you will need to change some of the environment variables in this script based on your system). The only requirement is to have a working `pytorch` installation.

**Note:** I could not get `torchcomms` to work with `torchrun` on SLURM yet.

### Supported features

**Backends:**
- `nccl`
- `ncclx` (see the `torchcomms` scripts)

**Communication Operations:**
- `all_reduce`
- `all_gather`
- `reduce_scatter`
- `broadcast`
- `reduce`

**Data Types (dtypes):**
- `torch.float32`
- `torch.float16`
- `torch.bfloat16`