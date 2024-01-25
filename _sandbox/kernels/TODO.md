Multi-Head-Attention CUDA:
- s0 = vanilla in CUDA

Skinning CUDA:
- s0 = full vertex_process kernel, per-vertex thread
- s1 = sharded vertex_process kernel, per-vertex thread
- s2 = two sharded kernels, bones + vertex_process, 2-vertex thread
- s3 = try tensor core

