# Building llama.cpp for NVIDIA Jetson

This guide covers building llama.cpp for NVIDIA Jetson devices (Orin, Xavier, TX2, Nano) with CUDA acceleration.

## Supported Devices

| Device | Compute Capability | Memory | Notes |
|--------|-------------------|--------|-------|
| Jetson Orin Nano | 8.7 | 8GB shared | Best performance/watt |
| Jetson Orin NX | 8.7 | 8-16GB shared | Production-ready |
| Jetson AGX Orin | 8.7 | 32-64GB shared | Highest performance |
| Jetson Xavier NX | 7.2 | 8-16GB shared | Good balance |
| Jetson AGX Xavier | 7.2 | 16-32GB shared | Mature platform |
| Jetson TX2 | 6.2 | 8GB shared | Legacy support |
| Jetson Nano | 5.3 | 4GB shared | Entry-level |

## Prerequisites

1. **JetPack SDK** - Install the appropriate JetPack version for your device:
   - Orin devices: JetPack 6.x (L4T R36.x)
   - Xavier devices: JetPack 5.x (L4T R35.x)
   - TX2/Nano: JetPack 4.x (L4T R32.x)

2. **CUDA Toolkit** - Included with JetPack, verify with:
   ```bash
   nvcc --version
   ```

3. **Build tools**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y cmake build-essential git
   ```

## Building with CUDA

### Critical: Disable CUDA Graphs

**IMPORTANT**: For Jetson Orin and other Tegra devices, you MUST disable CUDA graphs during build:

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Why Disable CUDA Graphs?

CUDA graphs provide performance benefits on discrete GPUs by recording and replaying sequences of CUDA operations. However, on Jetson's unified memory architecture:

1. **Graph compilation overhead** - The graph compilation can take longer than direct execution for the small batch sizes typical in inference
2. **Memory pressure** - Graph captures require additional memory for the recorded operations
3. **Stability issues** - Some graph operations behave differently on unified memory systems

The `GGML_CUDA_GRAPHS=OFF` flag is automatically applied for compute capability < 8.0, but explicitly setting it ensures consistent behavior on Orin (CC 8.7).

### Full Build Example (Jetson Orin)

```bash
# Clone the repository
git clone https://github.com/your-fork/llama.cpp.git
cd llama.cpp

# Configure for Jetson Orin with CUDA
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON

# Build (use all available cores)
cmake --build build --config Release -j$(nproc)

# Verify the build
./build/bin/llama-cli --version
```

### Architecture-Specific Builds

For optimal performance, specify the exact CUDA architecture:

```bash
# Jetson Orin (SM 8.7)
-DCMAKE_CUDA_ARCHITECTURES="87"

# Jetson Xavier (SM 7.2)
-DCMAKE_CUDA_ARCHITECTURES="72"

# Jetson TX2 (SM 6.2)
-DCMAKE_CUDA_ARCHITECTURES="62"

# Jetson Nano (SM 5.3)
-DCMAKE_CUDA_ARCHITECTURES="53"
```

## Installing Built Libraries for LlamaFarm

After building, copy the shared libraries to the LlamaFarm cache:

```bash
# Create the cache directory
CACHE_DIR="$HOME/.cache/llamafarm-llama/b7376"  # Use your build hash
mkdir -p "$CACHE_DIR"

# Copy required libraries
cp build/src/libllama.so "$CACHE_DIR/"
cp build/ggml/src/libggml.so* "$CACHE_DIR/"
cp build/ggml/src/libggml-base.so* "$CACHE_DIR/"
cp build/ggml/src/libggml-cpu.so* "$CACHE_DIR/"
cp build/ggml/src/libggml-cuda.so* "$CACHE_DIR/"
```

## Runtime Environment

### Environment Variables

For optimal Jetson performance, set these environment variables:

```bash
# Disable CUDA memory pool caching (reduces memory fragmentation)
export CUDA_CACHE_DISABLE=1

# Force synchronous inference (recommended for stability)
export LLAMAFARM_SYNC_INFERENCE=1

# Optional: Set thread count for CPU operations
export OMP_NUM_THREADS=4
```

### Memory Considerations

Jetson devices use unified memory (shared between CPU and GPU). Key considerations:

1. **Total memory is shared** - Running large models reduces available system RAM
2. **Use smaller context sizes** - Start with `n_ctx=2048` and increase if needed
3. **Monitor memory usage** - Use `tegrastats` to monitor memory
4. **KV cache quantization** - Use `cache_type_k=q4_0` to reduce memory by ~4x

### Recommended Model Settings for 8GB Jetson

```python
# Example configuration for memory-constrained Jetson
llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,              # Conservative context size
    n_gpu_layers=-1,         # Offload all layers to GPU
    n_batch=512,             # Reasonable batch size for 8GB
    flash_attn=True,         # Enable flash attention
    use_mmap=False,          # Disable mmap for unified memory
    cache_type_k="q4_0",     # Quantize KV cache keys
    cache_type_v="q4_0",     # Quantize KV cache values
)
```

## Troubleshooting

### "double free or corruption" crash

**Symptom**: Server crashes during startup with "double free or corruption" error.

**Cause**: CUDA backend initialization from worker thread on unified memory.

**Solution**: Ensure the llama.cpp backend is initialized from the main thread before any worker threads use it. In LlamaFarm, this is handled by `_init_llama_backend()` in `server.py`.

### Low performance (< 30 tok/s)

**Symptom**: Inference runs much slower than expected.

**Possible causes**:
1. **CUDA graphs enabled** - Rebuild with `GGML_CUDA_GRAPHS=OFF`
2. **Graph splits = 2** - Check logs for "graph splits = 2" indicating CPU/GPU split
3. **Insufficient GPU layers** - Ensure `n_gpu_layers=-1` to offload all layers
4. **Memory pressure** - Reduce `n_ctx` or use KV cache quantization

### Out of memory (OOM)

**Symptom**: "CUDA out of memory" or system becomes unresponsive.

**Solutions**:
1. Reduce `n_ctx` (context size)
2. Enable KV cache quantization (`cache_type_k="q4_0"`)
3. Use a smaller quantized model (Q4_K_M vs Q8_0)
4. Monitor with `tegrastats` to see actual memory usage

### Model loads on CPU instead of GPU

**Symptom**: Logs show "Using CPU" or no CUDA device detected.

**Solutions**:
1. Verify CUDA is properly installed: `nvcc --version`
2. Check for libggml-cuda.so in the library directory
3. Ensure JetPack is properly installed
4. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## Performance Benchmarks (Jetson Orin Nano 8GB)

With Qwen3-1.7B Q4_K_M and optimal settings:

| Metric | Value |
|--------|-------|
| Tokens/second | 35+ tok/s |
| Time to first token | ~140ms |
| Memory usage | ~2.5GB (model + KV cache) |
| Power consumption | ~15W |

## See Also

- [llama.cpp CUDA documentation](../backend/CUDA.md)
- [NVIDIA Jetson Developer Guide](https://developer.nvidia.com/embedded/learn/getting-started-jetson)
