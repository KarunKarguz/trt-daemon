#include "trt_server.hpp"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <atomic>
#include <signal.h>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <errno.h>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

// Standalone daemon that accepts inference requests over a Unix domain socket.


static std::atomic<bool> gStop{false};

static void on_signal(int) { gStop = true; }

/**
 * @brief Send exactly @p n bytes to @p fd, retrying short writes as needed.
 * @return 0 on success, -1 when the peer disconnects or a fatal socket error occurs.
 */
static int write_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t left = n;
  while (left) {
    ssize_t k = ::send(fd, p, left, MSG_NOSIGNAL);
    if (k < 0) {
      if (errno == EINTR) continue;
      return -1; // EPIPE/EAGAIN -> treat as error
    }
    p += static_cast<size_t>(k);
    left -= static_cast<size_t>(k);
  }
  return 0;
}

/**
 * @brief Blocking read that fills @p buf with @p n bytes, aborting on EOF or error.
 */
static int read_full(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t left = n;
  while (left) {
    ssize_t k = ::recv(fd, p, left, 0); // blocking read
    if (k == 0)  return -1;             // peer closed
    if (k < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    p += static_cast<size_t>(k);
    left -= static_cast<size_t>(k);
  }
  return 0;
}

/**
 * @brief Collects latency samples and prints a compact rolling summary.
 */
struct Stats {
  uint64_t count{0};
  double sum{0}, sumsq{0}, minv{1e9}, maxv{0};
  double p95{0};                 // simple percentile estimate (EMA)
  void add(double ms){
    count++; sum += ms; sumsq += ms*ms;
    if (ms < minv) minv = ms; if (ms > maxv) maxv = ms;
    // crude EMA for p95-ish tracking (fast + compact)
    double alpha = 0.05;
    if (ms > p95) p95 = (1-alpha)*p95 + alpha*ms;
    else          p95 = (1-alpha)*p95 + alpha*p95;
  }
  void print() const {
    double mean = sum / std::max<uint64_t>(1, count);
    double var  = (sumsq / std::max<uint64_t>(1, count)) - mean*mean;
    double stdv = var > 0 ? std::sqrt(var) : 0;
    std::cout << "[metrics] n=" << count
              << " mean=" << mean << "ms"
              << " p95~=" << p95 << "ms"
              << " min=" << minv << "ms"
              << " max=" << maxv << "ms"
              << " qps~=" << (mean>0 ? 1000.0/mean : 0)
              << "\n";
  }
};


/**
 * @brief Command-line knobs for choosing the engine file and socket path.
 */
struct Config {
  std::string engine = "model/resnet50_fp32.plan";
  std::string sock   = "/run/trt.sock";
  std::vector<std::pair<std::string, std::vector<int>>> shapes;
};

/**
 * @brief Parse CLI flags into a Config struct.
 */
static std::string trim(const std::string& s){
  auto begin = std::find_if_not(s.begin(), s.end(), [](unsigned char ch){ return std::isspace(ch); });
  auto end   = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch){ return std::isspace(ch); }).base();
  if (begin >= end) return std::string();
  return std::string(begin, end);
}

static bool parse_shape_spec(const std::string& spec, std::pair<std::string, std::vector<int>>& out, std::string& err){
  auto pos = spec.find('=');
  if (pos == std::string::npos){
    err = "shape spec '" + spec + "' missing '='";
    return false;
  }
  std::string name = trim(spec.substr(0, pos));
  std::string dimsStr = trim(spec.substr(pos + 1));
  if (name.empty()){
    err = "shape spec '" + spec + "' has empty binding name";
    return false;
  }
  if (dimsStr.empty()){
    err = "shape spec '" + spec + "' must include comma-separated dimensions";
    return false;
  }

  std::replace(dimsStr.begin(), dimsStr.end(), 'x', ',');
  std::replace(dimsStr.begin(), dimsStr.end(), 'X', ',');

  std::vector<int> dims;
  std::stringstream ss(dimsStr);
  std::string tok;
  while (std::getline(ss, tok, ',')){
    tok = trim(tok);
    if (tok.empty()) continue;
    try {
      int val = std::stoi(tok);
      if (val <= 0){
        err = "shape spec '" + spec + "' must use positive ints";
        return false;
      }
      dims.push_back(val);
    } catch (const std::exception&) {
      err = "shape spec '" + spec + "' contains non-integer dim";
      return false;
    }
  }
  if (dims.empty()){
    err = "shape spec '" + spec + "' must declare at least one dimension";
    return false;
  }
  out = {name, std::move(dims)};
  return true;
}

static void append_shape_list(const std::string& raw, std::vector<std::pair<std::string, std::vector<int>>>& dst){
  std::string normalized = raw;
  for (char& ch : normalized){
    if (ch == '\n' || ch == '\r') ch = ';';
  }

  size_t start = 0;
  while (start < normalized.size()){
    size_t end = normalized.find(';', start);
    std::string token = normalized.substr(start, end == std::string::npos ? std::string::npos : end - start);
    token = trim(token);
    if (!token.empty()){
      std::pair<std::string, std::vector<int>> spec;
      std::string err;
      if (!parse_shape_spec(token, spec, err)){
        fprintf(stderr, "shape parse error: %s\n", err.c_str());
        std::exit(1);
      }
      dst.push_back(std::move(spec));
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
}

static void append_shapes_from_env(Config& cfg){
  const char* env = std::getenv("TRT_SHAPES");
  if (!env || !*env) return;
  append_shape_list(env, cfg.shapes);
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  append_shapes_from_env(cfg);
  for (int i=1; i<argc; ++i) {
    std::string a = argv[i];
    if ((a=="-e"||a=="--engine") && i+1<argc) { cfg.engine = argv[++i]; }
    else if ((a=="-s"||a=="--sock") && i+1<argc) { cfg.sock = argv[++i]; }
    else if ((a=="-S"||a=="--shape") && i+1<argc) {
      std::pair<std::string, std::vector<int>> spec;
      std::string err;
      if (!parse_shape_spec(argv[++i], spec, err)){
        fprintf(stderr, "%s\n", err.c_str());
        std::exit(1);
      }
      cfg.shapes.push_back(std::move(spec));
    }
    else if (a=="-h" || a=="--help") {
      fprintf(stderr, "Usage: %s [-e engine.plan] [-s /path/socket] [--shape name=dim,dim,...]\n", argv[0]);
      std::exit(0);
    }
  }
  return cfg;
}

/**
 * @brief RAII helper for page-locked host buffers sized to TensorRT bindings.
 */
struct Pinned {
  void* p{nullptr}; size_t n{0};
  ~Pinned(){ if(p) cudaFreeHost(p); }
  void alloc(size_t bytes){
    if (p) cudaFreeHost(p);
    n = bytes;
    if (cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess)
      throw std::runtime_error("cudaHostAlloc failed");
  }
};

/**
 * @brief Render the daemon handshake that advertises binding metadata to clients.
 */
static std::string binding_header(const TRTServer& trt){
  std::ostringstream oss;
  oss << "TRTD 1\n";
  auto dump = [&oss](char tag, const Binding& b){
    oss << tag << ' ' << b.name << ' ' << b.bytes << ' ' << static_cast<int>(b.type) << ' ' << b.dims.nbDims;
    for (int i=0;i<b.dims.nbDims;i++) oss << ' ' << b.dims.d[i];
    oss << '\n';
  };
  for (const auto& b : trt.inputs()) dump('I', b);
  for (const auto& b : trt.outputs()) dump('O', b);
  oss << "END\n";
  return oss.str();
}

/**
 * @brief Create, bind, and listen on a Unix socket before handing it to epoll.
 */
static int make_server(const std::string& sockPath){
  ::unlink(sockPath.c_str());
  int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if(fd<0) { perror("socket"); std::exit(1); }
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path)-1);
  if(::bind(fd, (sockaddr*)&addr, sizeof(addr))<0){ perror("bind"); std::exit(1); }
  if(::listen(fd, 64)<0){ perror("listen"); std::exit(1); }
  ::chmod(sockPath.c_str(), 0666);
  return fd;
}

/**
 * @brief Main event loop: accepts clients, streams tensors, and runs inference.
 */
int main(int argc, char** argv){
    // signals: ignore SIGPIPE, catch INT/TERM for shutdown
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    Config cfg = parse_args(argc, argv);

    // ensure /run exists (some containers don’t have it)
    ::mkdir("/run", 0755);

    TRTServer::ShapeOverride shapeMap;
    for (const auto& spec : cfg.shapes) shapeMap[spec.first] = spec.second;

    if (!shapeMap.empty()){
        std::cout << "Applying runtime shapes:";
        for (const auto& kv : shapeMap){
            std::cout << ' ' << kv.first << '=';
            for (size_t i=0;i<kv.second.size();++i){
                std::cout << (i?"x":"") << kv.second[i];
            }
        }
        std::cout << "\n";
    }

    TRTServer trt(cfg.engine, shapeMap);

    std::vector<Pinned> hInputs(trt.inputs().size());
    std::vector<Pinned> hOutputs(trt.outputs().size());
    for (size_t i=0;i<trt.inputs().size(); ++i) hInputs[i].alloc(trt.inputs()[i].bytes);
    for (size_t i=0;i<trt.outputs().size(); ++i) hOutputs[i].alloc(trt.outputs()[i].bytes);
    std::vector<void*> inViews(trt.inputs().size());
    std::vector<void*> outViews(trt.outputs().size());

    std::string header = binding_header(trt);

    int sfd = make_server(cfg.sock);
    int ep  = ::epoll_create1(0);
    epoll_event ev{}; ev.events = EPOLLIN; ev.data.fd = sfd;
    ::epoll_ctl(ep, EPOLL_CTL_ADD, sfd, &ev);

    std::cout << "TRT daemon up, socket: " << cfg.sock << "  engine: " << cfg.engine << "\n";

    while (!gStop.load()){
        epoll_event events[16];
        int n = ::epoll_wait(ep, events, 16, 500); // 500ms timeout
        if (n < 0) {
            if (errno == EINTR) continue; // interrupted by signal
            perror("epoll_wait"); break;
        }
        for(int i=0;i<n;i++){
            int fd = events[i].data.fd;
            uint32_t evs = events[i].events;

            if (evs & (EPOLLHUP | EPOLLERR)) { ::close(fd); continue; }

            if (fd == sfd){
                // accept new
                int cfd = ::accept4(sfd, nullptr, nullptr, 0);
                if (cfd < 0) {
                    perror("accept4");
                    continue;
                }
                if (cfd >= 0){
                    if (write_all(cfd, header.data(), header.size()) != 0){
                        ::close(cfd);
                        continue;
                    }
                    epoll_event cev{}; cev.events = EPOLLIN; cev.data.fd = cfd;
                    ::epoll_ctl(ep, EPOLL_CTL_ADD, cfd, &cev);
                }
            } else {
                auto t0 = std::chrono::high_resolution_clock::now();

                bool io_ok = true;
                for (size_t i = 0; i < hInputs.size(); ++i) {
                    if (read_full(fd, hInputs[i].p, hInputs[i].n) != 0) { io_ok = false; break; }
                    inViews[i] = hInputs[i].p;
                }
                if (!io_ok) { ::close(fd); continue; }

                for (size_t i = 0; i < hOutputs.size(); ++i) {
                    outViews[i] = hOutputs[i].p;
                }

                try { trt.infer(inViews, outViews); }
                catch (const std::exception& e) { std::cerr << "infer error: " << e.what() << "\n"; ::close(fd); continue; }

                bool write_ok = true;
                for (size_t i = 0; i < hOutputs.size(); ++i) {
                    if (write_all(fd, hOutputs[i].p, hOutputs[i].n) != 0) { write_ok = false; break; }
                }
                if (!write_ok) { ::close(fd); continue; }

                double ms = std::chrono::duration<double,std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
                static Stats S;
                S.add(ms);
                if (S.count % 100 == 0) S.print();   // print every 100 requests
            }
        }
    }
    // shutdown path
    ::close(sfd);
    ::close(ep);
    // unlink in case it still exists
    ::unlink(cfg.sock.c_str());
    std::cout << "TRT daemon stopped.\n";
    return 0;
}
