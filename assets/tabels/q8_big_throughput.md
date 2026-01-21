# Question 8 - BIG model throughput benchmark (GPU)

Generated from Slurm `.out` files.

| Batch size | Throughput (img/s) | Elapsed (s) | Total images | VRAM max alloc (MB) | VRAM max reserv (MB) | Error |
|-----------:|-------------------:|------------:|------------:|--------------------:|---------------------:|:------|
| 8 | 844.36 | 0.4737 | 400 | 114.90 | 142.00 |  |
| 16 | 1052.27 | 0.7603 | 800 | 131.55 | 162.00 |  |
| 32 | 1184.65 | 1.3506 | 1600 | 160.93 | 196.00 |  |
| 64 | 1298.34 | 2.4647 | 3200 | 222.80 | 266.00 |  |
| 128 | 1642.73 | 3.8960 | 6400 | 346.55 | 482.00 |  |
| 256 | 1699.95 | 7.5296 | 12800 | 595.05 | 870.00 |  |
| 512 | 1758.28 | 14.5597 | 25600 | 1089.05 | 1616.00 |  |

## Metadata

- Model params: 23512130
- Model size (fp32 params only): 89.69 MB
- GPU: NVIDIA A100-SXM4-40GB MIG 1g.5gb
