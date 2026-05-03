### `nccl`/`ncclx` benchmark

Run a quick test to benchmark the speed of your `pytorch` + `nccl`/`ncclx` setup for distributed training. Mostly written by Gemini 3.1 Pro.

### Supported features

**Libraries:**
- `torch.distributed` (use [`test_distributed.py`](test_distributed.py)/[`test_distributed.sh`](test_distributed.sh))
- `torchcomms` (use [`test_torchcomms.py`](test_torchcomms.py)/[`test_torchcomms.sh`](test_torchcomms.sh))

**Backends:**
- `nccl`
- `ncclx` (only available in `torchcomms`)

**Collective ops:**
- `all_reduce`
- `all_gather`
- `reduce_scatter`
- `broadcast`
- `reduce`

**Data types (dtypes):**
- `torch.float32`
- `torch.float16`
- `torch.bfloat16`

### Requirements

The only requirement is to have a working PyTorch installation. If you would like to test the [`torchcomms`](https://github.com/meta-pytorch/torchcomms) library, you will need to have that installed as well (see the instructions in the link). 

### Glitches

- I could not get `torchcomms` to work properly with `torchrun` on SLURM yet.

