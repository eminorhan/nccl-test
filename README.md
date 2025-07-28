### nccl `dist.all_reduce()` bechmark

A quick nccl test using `dist.all_reduce()` to benchmark your `pytorch` + `nccl` setup for distributed training. Use the SLURM batch script [nccl_test.sh](nccl_test.sh) to run this benchmark on a SLURM cluster (you will need to change some of the environment variables in this script based on your system).