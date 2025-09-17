#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>

struct CudaStream {
  cudaStream_t s{nullptr};
  CudaStream();
  ~CudaStream();
};

class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity s, const char* msg) noexcept override;
};

struct Binding {
  std::string name;
  int index{-1};
  nvinfer1::Dims dims{};
  size_t bytes{0};
  bool isInput{false};
};

class TRTServer {
public:
  explicit TRTServer(const std::string& enginePath);
  ~TRTServer();

  // reshape if dynamic (optional: set -1 dims here)
  void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

  // Synchronous infer: copies H->D, enqueue, D->H
  // hInput/hOutput must be pinned buffers sized to bindings’ bytes
  void infer(void* hInput, void* hOutput);

  const Binding& input()  const { return inputB_; }
  const Binding& output() const { return outputB_; }

private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
  CudaStream stream_;

  Binding inputB_{}, outputB_{};
  void *dInput_{nullptr}, *dOutput_{nullptr};

  void loadEngineFromFile(const std::string& path);
  static size_t vol(const nvinfer1::Dims& d);
  static size_t eltSize(nvinfer1::DataType t);
  void allocDevice();
};
