#include "trt_server.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>

// Implementation for the thin TensorRT execution wrapper declared in the header.

using namespace nvinfer1;

CudaStream::CudaStream(){ cudaStreamCreate(&s); }
CudaStream::~CudaStream(){ if (s) cudaStreamDestroy(s); }

void TRTLogger::log(Severity sev, const char* msg) noexcept {
  if (sev <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
}


size_t TRTServer::vol(const Dims& d){
  size_t v = 1;
  for(int i=0;i<d.nbDims;i++){
    int dim = d.d[i];
    if (dim <= 0) return 0; // bail early; runtime shape must be provided via setInputShape
    v *= static_cast<size_t>(dim);
  }
  return v;
}
size_t TRTServer::eltSize(DataType t){
  switch(t){
    case DataType::kFLOAT: return 4;
    case DataType::kHALF:  return 2;
    case DataType::kINT32: return 4;
    case DataType::kINT8:  return 1;
    case DataType::kBOOL:  return 1;
#if TENSORRT_VERSION >= 8500
    case DataType::kFP8:   return 1; // not used on Maxwell
#endif
    default: return 4;
  }
}

void TRTServer::loadEngineFromFile(const std::string& path){
  std::ifstream f(path, std::ios::binary);
  if(!f) throw std::runtime_error("engine file not found: " + path);
  f.seekg(0, std::ios::end);
  size_t sz = f.tellg(); f.seekg(0, std::ios::beg);
  std::vector<char> buf(sz);
  f.read(buf.data(), sz);

  runtime_.reset(createInferRuntime(logger_));
  if(!runtime_) throw std::runtime_error("createInferRuntime failed");

  engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
  if(!engine_) throw std::runtime_error("deserializeCudaEngine failed");

  ctx_.reset(engine_->createExecutionContext());
  if(!ctx_) throw std::runtime_error("createExecutionContext failed");
}

TRTServer::TRTServer(const std::string& enginePath, const ShapeOverride& runtimeShapes){
  loadEngineFromFile(enginePath);

  bindings_.clear();

  if (engine_->getNbOptimizationProfiles() > 0) {
    if (!ctx_->setOptimizationProfile(0))
      throw std::runtime_error("setOptimizationProfile(0) failed");
  }

  applyRuntimeShapes(runtimeShapes);
  collectBindings();
  allocDevice();
}

void TRTServer::allocDevice(){
  dInputs_.resize(inputs_.size(), nullptr);
  dOutputs_.resize(outputs_.size(), nullptr);

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].bytes == 0)
      throw std::runtime_error("input binding has zero-sized tensor (provide runtime shape with --shape or TRT_SHAPES)");
    if (cudaMalloc(&dInputs_[i], inputs_[i].bytes) != cudaSuccess)
      throw std::runtime_error("cudaMalloc input binding failed");
    bindings_[inputs_[i].index] = dInputs_[i];
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].bytes == 0)
      throw std::runtime_error("output binding has zero-sized tensor (provide runtime shape with --shape or TRT_SHAPES)");
    if (cudaMalloc(&dOutputs_[i], outputs_[i].bytes) != cudaSuccess)
      throw std::runtime_error("cudaMalloc output binding failed");
    bindings_[outputs_[i].index] = dOutputs_[i];
  }
}

TRTServer::~TRTServer(){
  for (void* ptr : dInputs_) if (ptr) cudaFree(ptr);
  for (void* ptr : dOutputs_) if (ptr) cudaFree(ptr);
}

void TRTServer::setInputShape(const std::string& name, const Dims& d){
  int idx = engine_->getBindingIndex(name.c_str());
  if (idx < 0) throw std::runtime_error("bad binding name");
  if (!ctx_->setBindingDimensions(idx, d)) throw std::runtime_error("setBindingDimensions failed");
}

nvinfer1::Dims TRTServer::toDims(const std::vector<int>& dimsVec, const std::string& name){
  if (dimsVec.empty())
    throw std::runtime_error("shape override for '" + name + "' must include at least one dimension");
  if (dimsVec.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    throw std::runtime_error("shape override for '" + name + "' exceeds TensorRT max dims");
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(dimsVec.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    int value = dimsVec[static_cast<size_t>(i)];
    if (value <= 0)
      throw std::runtime_error("shape override for '" + name + "' must use positive dims");
    dims.d[i] = value;
  }
  return dims;
}

void TRTServer::applyRuntimeShapes(const ShapeOverride& runtimeShapes){
  for (const auto& kv : runtimeShapes) {
    const std::string& name = kv.first;
    int idx = engine_->getBindingIndex(name.c_str());
    if (idx < 0)
      throw std::runtime_error("shape override refers to unknown binding: " + name);
    if (!engine_->bindingIsInput(idx))
      throw std::runtime_error("shape override '" + name + "' targets an output binding");
    nvinfer1::Dims dims = toDims(kv.second, name);
    if (!ctx_->setBindingDimensions(idx, dims))
      throw std::runtime_error("setBindingDimensions failed for binding: " + name);
  }

  if (!ctx_->allInputDimensionsSpecified()) {
    throw std::runtime_error("not all dynamic input dimensions were specified; provide --shape or TRT_SHAPES overrides");
  }
}

void TRTServer::collectBindings(){
  inputs_.clear();
  outputs_.clear();
  bindings_.assign(engine_->getNbBindings(), nullptr);

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    if (!engine_->isExecutionBinding(i))
      continue;

    Binding b;
    b.name = engine_->getBindingName(i);
    b.index = i;
    b.isInput = engine_->bindingIsInput(i);
    b.type = engine_->getBindingDataType(i);
    b.dims = ctx_->getBindingDimensions(i);

    size_t volBytes = vol(b.dims);
    if (volBytes == 0) {
      std::ostringstream oss;
      oss << "binding '" << b.name << "' has unresolved dimensions";
      throw std::runtime_error(oss.str());
    }
    b.bytes = volBytes * eltSize(b.type);

    if (b.isInput) inputs_.push_back(b); else outputs_.push_back(b);
  }

  if (inputs_.empty())
    throw std::runtime_error("engine exposes no input execution bindings");
  if (outputs_.empty())
    throw std::runtime_error("engine exposes no output execution bindings");
}

void TRTServer::infer(const std::vector<void*>& hInputs, const std::vector<void*>& hOutputs){
  if (hInputs.size() != inputs_.size())
    throw std::runtime_error("input buffer count mismatch");
  if (hOutputs.size() != outputs_.size())
    throw std::runtime_error("output buffer count mismatch");

  for (size_t i = 0; i < inputs_.size(); ++i) {
    cudaMemcpyAsync(dInputs_[i], hInputs[i], inputs_[i].bytes, cudaMemcpyHostToDevice, stream_.s);
  }

  if (!ctx_->enqueueV2(bindings_.data(), stream_.s, nullptr))
    throw std::runtime_error("enqueueV2 failed");

  for (size_t i = 0; i < outputs_.size(); ++i) {
    cudaMemcpyAsync(hOutputs[i], dOutputs_[i], outputs_[i].bytes, cudaMemcpyDeviceToHost, stream_.s);
  }
  cudaStreamSynchronize(stream_.s);
}
