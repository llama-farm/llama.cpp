#!/bin/bash
#
# Build llama.cpp for NVIDIA Jetson devices
#
# This script builds llama.cpp with CUDA support optimized for Jetson's
# unified memory architecture. Key features:
#   - Disables CUDA graphs (required for stability on Tegra)
#   - Sets appropriate CUDA architecture for the device
#   - Builds shared libraries for LlamaFarm integration
#
# Usage:
#   ./scripts/build-jetson.sh [device]
#
# Devices:
#   orin      - Jetson Orin (Nano, NX, AGX) - SM 8.7 (default)
#   xavier    - Jetson Xavier (NX, AGX) - SM 7.2
#   tx2       - Jetson TX2 - SM 6.2
#   nano      - Jetson Nano - SM 5.3
#
# Examples:
#   ./scripts/build-jetson.sh           # Build for Orin (default)
#   ./scripts/build-jetson.sh xavier    # Build for Xavier
#
# Output:
#   Libraries are built in ./build/ directory
#   Copy to ~/.cache/llamafarm-llama/<hash>/ for LlamaFarm use
#

set -e

DEVICE="${1:-orin}"

# Map device names to CUDA architectures
case "$DEVICE" in
    orin|orin-nano|orin-nx|agx-orin)
        CUDA_ARCH="87"
        DEVICE_NAME="Jetson Orin"
        ;;
    xavier|xavier-nx|agx-xavier)
        CUDA_ARCH="72"
        DEVICE_NAME="Jetson Xavier"
        ;;
    tx2)
        CUDA_ARCH="62"
        DEVICE_NAME="Jetson TX2"
        ;;
    nano)
        CUDA_ARCH="53"
        DEVICE_NAME="Jetson Nano"
        ;;
    *)
        echo "Unknown device: $DEVICE"
        echo "Supported: orin, xavier, tx2, nano"
        exit 1
        ;;
esac

echo "=============================================="
echo "Building llama.cpp for $DEVICE_NAME (SM $CUDA_ARCH)"
echo "=============================================="
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit (part of JetPack)."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "CUDA version: $CUDA_VERSION"

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Install with: sudo apt-get install cmake"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
echo "CMake version: $CMAKE_VERSION"
echo ""

# Create build directory
BUILD_DIR="build-jetson-${DEVICE}"
echo "Build directory: $BUILD_DIR"
echo ""

# Configure with CMake
echo "Configuring CMake..."
cmake -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DBUILD_SHARED_LIBS=ON

echo ""

# Build
NPROC=$(nproc)
echo "Building with $NPROC threads..."
cmake --build "$BUILD_DIR" --config Release -j"$NPROC"

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Libraries built in: $BUILD_DIR/"
echo ""
echo "Key files:"
echo "  $BUILD_DIR/src/libllama.so"
echo "  $BUILD_DIR/ggml/src/libggml*.so"
echo ""
echo "To use with LlamaFarm, copy libraries to cache:"
echo ""
echo "  CACHE_DIR=\"\$HOME/.cache/llamafarm-llama/\$(date +%s)\""
echo "  mkdir -p \"\$CACHE_DIR\""
echo "  cp $BUILD_DIR/src/libllama.so \"\$CACHE_DIR/\""
echo "  cp $BUILD_DIR/ggml/src/libggml*.so* \"\$CACHE_DIR/\""
echo ""
echo "Then set LLAMAFARM_LLAMA_LIB_DIR to point to the cache directory."
echo ""

# Verify build
if [ -f "$BUILD_DIR/bin/llama-cli" ]; then
    echo "Verifying build..."
    "$BUILD_DIR/bin/llama-cli" --version 2>/dev/null || true
fi
