#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

/**
 * @brief RAII wrapper for CUDA streams used by the TensorRT context.
 */
struct CudaStream {
  cudaStream_t s{nullptr};
  CudaStream();
  ~CudaStream();
};

/**
 * @brief Thin TensorRT logger that mirrors warnings (and higher severity) to stderr.
 */
class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity s, const char* msg) noexcept override;
};

/**
 * @brief Captures metadata for a single TensorRT binding exposed by the engine.
 */
struct Binding {
  std::string name;                        ///< Human-readable binding identifier from the engine
  int index{-1};                           ///< Binding index inside the TensorRT engine
  nvinfer1::Dims dims{};                   ///< Declared tensor dimensions (may include dynamic entries)
  nvinfer1::DataType type{nvinfer1::DataType::kFLOAT}; ///< Element data type
  size_t bytes{0};                         ///< Total byte size for a fully-resolved tensor
  bool isInput{false};                     ///< True if the binding is an input tensor
};

struct RuntimeDeleter {
  void operator()(nvinfer1::IRuntime* p) const noexcept { if (p) p->destroy(); }
};

struct EngineDeleter {
  void operator()(nvinfer1::ICudaEngine* p) const noexcept { if (p) p->destroy(); }
};

struct ContextDeleter {
  void operator()(nvinfer1::IExecutionContext* p) const noexcept { if (p) p->destroy(); }
};

/**
 * @brief Convenience wrapper that loads a serialized TensorRT engine and exposes synchronous inference.
 */
class TRTServer {
public:
  using ShapeOverride = std::unordered_map<std::string, std::vector<int>>;

  /**
   * @param enginePath Filesystem path to a TensorRT `.plan` (serialized engine) file.
   * @throws std::runtime_error when the engine cannot be read or deserialized.
   */
  explicit TRTServer(const std::string& enginePath, const ShapeOverride& runtimeShapes = {});
  ~TRTServer();

  /**
   * @brief Override an input binding shape for engines that use dynamic dimensions.
   *
   * Calling this before the first inference lets callers specify the runtime shape for a named binding.
   * @param name Binding name exactly as reported by TensorRT.
   * @param dims Concrete dimensions to apply. TensorRT expects fully-resolved values where dynamic slots are set.
   */
  void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

  /**
   * @brief Perform a synchronous inference against the loaded engine.
   *
   * Host buffers must remain valid for the duration of the call and should be page-locked for best performance.
   * Buffers must be sized to match the `bytes` field advertised in the corresponding binding metadata.
   *
   * @param hInputs Host memory pointers for each input binding in declaration order.
   * @param hOutputs Host memory pointers for each output binding in declaration order.
   */
  void infer(const std::vector<void*>& hInputs, const std::vector<void*>& hOutputs);

  /** @return Metadata describing each input binding in declaration order. */
  const std::vector<Binding>& inputs() const { return inputs_; }
  /** @return Metadata describing each output binding in declaration order. */
  const std::vector<Binding>& outputs() const { return outputs_; }

private:
  TRTLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext, ContextDeleter> ctx_{nullptr};
  CudaStream stream_;

  std::vector<Binding> inputs_;
  std::vector<Binding> outputs_;
  std::vector<void*> dInputs_;
  std::vector<void*> dOutputs_;
  std::vector<void*> bindings_;

  /** Load and deserialize a serialized engine into runtime_/engine_/ctx_. */
  void loadEngineFromFile(const std::string& path);
  /** Apply caller-provided runtime shapes before materialising bindings. */
  void applyRuntimeShapes(const ShapeOverride& runtimeShapes);
  /** Populate binding metadata after shapes are fixed. */
  void collectBindings();
  /** Compute the flattened volume for a TensorRT dimension declaration. */
  static size_t vol(const nvinfer1::Dims& d);
  /** Map a TensorRT data type enum to its byte width. */
  static size_t eltSize(nvinfer1::DataType t);
  /** Convert a vector<int> into a TensorRT Dims struct with validation. */
  static nvinfer1::Dims toDims(const std::vector<int>& dimsVec, const std::string& name);
  /** Allocate device buffers that mirror the engine binding layout. */
  void allocDevice();
};
