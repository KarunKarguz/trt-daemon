#include "trt_server.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

// Implementation for the thin TensorRT execution wrapper declared in the header.

using namespace nvinfer1;

CudaStream::CudaStream(){ cudaStreamCreate(&s); }
CudaStream::~CudaStream(){ if (s) cudaStreamDestroy(s); }

void TRTLogger::log(Severity sev, const char* msg) noexcept {
  if (sev <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
}

static auto delRuntime = [](IRuntime* p){ if (p) p->destroy(); };
static auto delEngine  = [](ICudaEngine* p){ if (p) p->destroy(); };
static auto delCtx     = [](IExecutionContext* p){ if (p) p->destroy(); };

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

TRTServer::TRTServer(const std::string& enginePath){
  loadEngineFromFile(enginePath);

  inputs_.clear();
  outputs_.clear();
  bindings_.resize(engine_->getNbBindings());

  for (int i=0;i<engine_->getNbBindings();++i){
    Binding b;
    b.name   = engine_->getBindingName(i);
    b.index  = i;
    b.isInput= engine_->bindingIsInput(i);
    auto t   = engine_->getBindingDataType(i);
    auto d   = engine_->getBindingDimensions(i);
    b.dims   = d;
    b.type   = t;
    b.bytes  = vol(d) * eltSize(t);
    if (b.isInput) inputs_.push_back(b); else outputs_.push_back(b);
  }
  allocDevice();
}

void TRTServer::allocDevice(){
  dInputs_.resize(inputs_.size(), nullptr);
  dOutputs_.resize(outputs_.size(), nullptr);

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].bytes == 0)
      throw std::runtime_error("input binding has zero-sized tensor (unsupported dynamic dims)");
    if (cudaMalloc(&dInputs_[i], inputs_[i].bytes) != cudaSuccess)
      throw std::runtime_error("cudaMalloc input binding failed");
    bindings_[inputs_[i].index] = dInputs_[i];
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs_[i].bytes == 0)
      throw std::runtime_error("output binding has zero-sized tensor (unsupported dynamic dims)");
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
