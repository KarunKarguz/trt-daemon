#include <signal.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <cmath>

static const char* kSock = "/run/trt.sock";

static inline double ms_since(const std::chrono::high_resolution_clock::time_point& t0) {
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
}

// send exactly n bytes (returns 0 on success, -1 on error)
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

// read exactly n bytes (returns 0 on success, -1 on error/EOF)
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

  // Match your ResNet50 engine (1x3x224x224 input, 1000 outputs), FP32
  const size_t in_elems  = 1*3*224*224;
  const size_t out_elems = 1000;
  const size_t in_bytes  = in_elems  * sizeof(float);
  const size_t out_bytes = out_elems * sizeof(float);

  std::vector<float> in(in_elems, 0.5f), out(out_elems);

  // warmup
  for (int i=0;i<warm;i++){
    if (write_all(fd, in.data(), in_bytes) != 0) { std::perror("write_all"); return 2; }
    if (read_full(fd, out.data(), out_bytes) != 0) { std::perror("read_full"); return 2; }
  }

  // timed loop
  double sum = 0.0, sumsq = 0.0, minms = 1e9, maxms = 0.0;
  for (int i=0;i<iters;i++){
    auto t0 = std::chrono::high_resolution_clock::now();

    if (write_all(fd, in.data(), in_bytes) != 0) { std::perror("write_all"); return 2; }
    if (read_full(fd, out.data(), out_bytes) != 0) { std::perror("read_full"); return 2; }

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
  std::cout << "Sample out[0]=" << out[0] << "\n";
  return 0;
}
