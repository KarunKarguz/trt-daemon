rm -rf build
cmake -B build -S . -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build -j$(nproc)