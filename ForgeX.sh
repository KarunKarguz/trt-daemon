rm -rf build
mkdir -p build && cd build
cmake -DCUDAToolkit_ROOT=/usr/local/cuda ..
make -j$(nproc)
cd .. 