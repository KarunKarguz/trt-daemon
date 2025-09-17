# GitLab CI Checklist

The pipeline baked into `.gitlab-ci.yml` keeps the project honest without over-engineering things.

## Runners
- Use a shared GitLab runner if GPU is not required; the build stage only needs headers and compilers.
- For GPU-enabled smoke tests, tag a dedicated runner with CUDA drivers and mount `/dev/nvidia*`. That can come later once the repository is public.

## Required variables
- `CI_REGISTRY_USER` / `CI_REGISTRY_PASSWORD` – only needed if you decide to push container images.
- `NVIDIA_VISIBLE_DEVICES` – set to `all` for GPU jobs, or leave empty for CPU-only builds.

## Job overview
- `build`: compiles the daemon with CMake against `/usr/local/cuda`.
- `smoke-test`: executes the client binary with low iteration counts. In a CPU-only environment it exits non-zero because CUDA context creation fails, so the pipeline lets it run and ignores failures (`|| true`). For real GPU runners remove that guard.

## Artifacts and caching
- Artifacts keep the fresh binaries for a week; tweak to match your release cadence.
- If you want faster incremental builds, add:
  ```yaml
  cache:
    key: "cmake-${CI_COMMIT_REF_SLUG}"
    paths:
      - build/
  ```
  Remember to clear the cache when upgrading TensorRT or CUDA.

## Security review
- Ensure no proprietary models or logs are published.
- Run `cmake --build build --target clean` before tagging a release to avoid stale binaries in Git.
