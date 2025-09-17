#include "trt_server.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

using namespace nvinfer1;

CudaStream::CudaStream(){ cudaStreamCreate(&s); }
CudaStream::~CudaStream(){ if (s) cudaStreamDestroy(s); }

void TRTLogger::log(Severity sev, const char* msg) noexcept {
  if (sev <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
}

static auto delRuntime = [](IRuntime* p){ if (p) p->destroy(); };
static auto delEngine  = [](ICudaEngine* p){ if (p) p->destroy(); };
static auto delCtx     = [](IExecutionContext* p){ if (p) p->destroy(); };

size_t TRTServer::vol(const Dims& d){ size_t v = 1; for(int i=0;i<d.nbDims;i++) v*= d.d[i]; return v; }
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

  // Assume exactly 1 input + 1 output for brevity (extend as needed)
  for (int i=0;i<engine_->getNbBindings();++i){
    Binding b;
    b.name   = engine_->getBindingName(i);
    b.index  = i;
    b.isInput= engine_->bindingIsInput(i);
    auto t   = engine_->getBindingDataType(i);
    auto d   = engine_->getBindingDimensions(i);
    b.dims   = d;
    b.bytes  = vol(d) * eltSize(t);

    if (b.isInput) inputB_ = b; else outputB_ = b;
  }
  allocDevice();
}

void TRTServer::allocDevice(){
  if (cudaMalloc(&dInput_,  inputB_.bytes)  != cudaSuccess) throw std::runtime_error("cudaMalloc dInput");
  if (cudaMalloc(&dOutput_, outputB_.bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc dOutput");
}

TRTServer::~TRTServer(){
  if (dInput_)  cudaFree(dInput_);
  if (dOutput_) cudaFree(dOutput_);
}

void TRTServer::setInputShape(const std::string& name, const Dims& d){
  int idx = engine_->getBindingIndex(name.c_str());
  if (idx < 0) throw std::runtime_error("bad binding name");
  if (!ctx_->setBindingDimensions(idx, d)) throw std::runtime_error("setBindingDimensions failed");
}

void TRTServer::infer(void* hInput, void* hOutput){
  // H->D (async)
  cudaMemcpyAsync(dInput_, hInput, inputB_.bytes, cudaMemcpyHostToDevice, stream_.s);

  // enqueue
  void* bindings[] = { dInput_, dOutput_ };
  if (!ctx_->enqueueV2(bindings, stream_.s, nullptr))
    throw std::runtime_error("enqueueV2 failed");

  // D->H (async)
  cudaMemcpyAsync(hOutput, dOutput_, outputB_.bytes, cudaMemcpyDeviceToHost, stream_.s);
  cudaStreamSynchronize(stream_.s);
}
