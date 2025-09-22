#include <signal.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>

// Benchmark client that connects to the daemon, parses the handshake, and exercises inference.

static const char* kSock = "/run/trt.sock";

/**
 * @brief Lightweight mirror of the binding metadata exposed by the daemon handshake.
 */
struct BindingInfo {
  std::string name;
  size_t bytes{0};
  int dtype{0};
  std::vector<int> dims;
};

static inline double ms_since(const std::chrono::high_resolution_clock::time_point& t0) {
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
}

/**
 * @brief Write @p n bytes to @p fd, retrying until completion unless the peer fails.
 */
static int write_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t left = n;
  while (left) {
    ssize_t k = ::send(fd, p, left, MSG_NOSIGNAL);  // suppress SIGPIPE
    if (k < 0) {
      if (errno == EINTR) continue;                 // retry on signal
      return -1;                                    // EPIPE/EAGAIN -> treat as error
    }
    p += static_cast<size_t>(k);
    left -= static_cast<size_t>(k);
  }
  return 0;
}

/**
 * @brief Blocking read helper that returns -1 when the peer closes the socket or errors out.
 */
static int read_full(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t left = n;
  while (left) {
    ssize_t k = ::recv(fd, p, left, 0);
    if (k == 0)  return -1;                         // peer closed
    if (k < 0) {
      if (errno == EINTR) continue;                 // retry on signal
      return -1;                                    // other error
    }
    p += static_cast<size_t>(k);
    left -= static_cast<size_t>(k);
  }
  return 0;
}

/**
 * @brief Read a single newline-terminated line from the socket.
 */
static bool read_line(int fd, std::string& out){
  out.clear();
  char ch;
  while (true){
    ssize_t k = ::recv(fd, &ch, 1, 0);
    if (k == 0) return false;
    if (k < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (ch == '\n') break;
    out.push_back(ch);
  }
  return true;
}

/**
 * @brief Consume the daemon header and populate the input/output binding lists.
 */
static bool parse_header(int fd, std::vector<BindingInfo>& inputs, std::vector<BindingInfo>& outputs){
  std::string line;
  if (!read_line(fd, line)) return false;
  if (line != "TRTD 1") return false;
  while (read_line(fd, line)){
    if (line == "END") break;
    if (line.empty()) continue;
    std::istringstream iss(line);
    char kind;
    BindingInfo info;
    int ndims = 0;
    iss >> kind >> info.name >> info.bytes >> info.dtype >> ndims;
    if (!iss) return false;
    info.dims.resize(ndims);
    for (int i=0;i<ndims;i++) iss >> info.dims[i];
    if (!iss) return false;
    if (kind == 'I') inputs.push_back(info);
    else if (kind == 'O') outputs.push_back(info);
  }
  return !inputs.empty() && !outputs.empty();
}

/**
 * @brief Command-line entrypoint that runs a timed inference loop against the daemon.
 */
int main(int argc, char** argv){
  // be robust to broken pipes
  signal(SIGPIPE, SIG_IGN);

  int iters = (argc > 1) ? std::atoi(argv[1]) : 200;   // default: 200 inferences
  int warm  = (argc > 2) ? std::atoi(argv[2]) : 20;    // default: 20 warmup

  // connect
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) { perror("socket"); return 1; }
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, kSock, sizeof(addr.sun_path)-1);
  if(::connect(fd, (sockaddr*)&addr, sizeof(addr))<0){ perror("connect"); return 1; }

  std::vector<BindingInfo> inputs;
  std::vector<BindingInfo> outputs;
  if (!parse_header(fd, inputs, outputs)){
    std::cerr << "failed to parse server header" << std::endl;
    return 1;
  }

  std::vector<std::vector<uint8_t>> inBuffers(inputs.size());
  for (size_t i=0;i<inputs.size();++i) inBuffers[i].assign(inputs[i].bytes, 0);
  std::vector<std::vector<uint8_t>> outBuffers(outputs.size());
  for (size_t i=0;i<outputs.size();++i) outBuffers[i].resize(outputs[i].bytes);

  // seed dummy data for benchmarking (float32 only for now)
  for (size_t i=0;i<inputs.size();++i){
    if (inputs[i].dtype == 0) {
      auto* view = reinterpret_cast<float*>(inBuffers[i].data());
      size_t count = inputs[i].bytes / sizeof(float);
      for (size_t k=0;k<count;k++) view[k] = 0.5f;
    }
  }

  // warmup
  for (int i=0;i<warm;i++){
    for (size_t i=0;i<inputs.size();++i){
      if (write_all(fd, inBuffers[i].data(), inputs[i].bytes) != 0) { std::perror("write_all"); return 2; }
    }
    for (size_t i=0;i<outputs.size();++i){
      if (read_full(fd, outBuffers[i].data(), outputs[i].bytes) != 0) { std::perror("read_full"); return 2; }
    }
  }

  // timed loop
  double sum = 0.0, sumsq = 0.0, minms = 1e9, maxms = 0.0;
  for (int i=0;i<iters;i++){
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t j=0;j<inputs.size();++j){
      if (write_all(fd, inBuffers[j].data(), inputs[j].bytes) != 0) { std::perror("write_all"); return 2; }
    }
    for (size_t j=0;j<outputs.size();++j){
      if (read_full(fd, outBuffers[j].data(), outputs[j].bytes) != 0) { std::perror("read_full"); return 2; }
    }

    double dt = ms_since(t0);
    sum += dt; sumsq += dt*dt;
    if (dt < minms) minms = dt;
    if (dt > maxms) maxms = dt;
  }
  ::close(fd);

  double mean = sum / iters;
  double var  = (sumsq/iters) - (mean*mean);
  double stdv = var > 0 ? std::sqrt(var) : 0;
  double qps  = mean > 0 ? 1000.0 / mean : 0.0;

  std::cout << "Iters=" << iters
            << "  mean=" << mean << " ms"
            << "  std="  << stdv << " ms"
            << "  min="  << minms << " ms"
            << "  max="  << maxms << " ms"
            << "  ~QPS=" << qps   << "\n";
  if (!outputs.empty() && outputs[0].dtype == 0) {
    float sample = reinterpret_cast<const float*>(outBuffers[0].data())[0];
    std::cout << "Sample out[0]=" << sample << "\n";
  }
  return 0;
}
