cmake_minimum_required(VERSION 3.18)
project(trtdaemon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

include_directories(/usr/include/x86_64-linux-gnu)

add_executable(trt-daemon
  src/daemon.cpp
  src/trt_server.cpp
)

target_compile_options(trt-daemon PRIVATE -Wno-deprecated-declarations)

target_include_directories(trt-daemon PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  include
)

target_link_libraries(trt-daemon PRIVATE
  nvinfer
  CUDA::cudart
  Threads::Threads
)

add_executable(trt-client src/client.cpp)
