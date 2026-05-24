---
name: Bug report
about: Something is broken — help us fix it
title: "[bug] "
labels: bug
---

**Describe the bug**
What happened vs. what you expected.

**Environment**
- DGX Spark model / GPU arch:
- Phase / optimization enabled: (baseline, int8-lm-head, mtp, hybrid, all)
- Docker image tag (from `docker images`):
- vLLM version (from `/workspace/build-metadata.yaml` inside the container):

**Reproduction steps**
1. …
2. …
3. …

**Logs**
```
paste relevant `docker logs vllm-qwen36b` output
```

**Benchmark output (if applicable)**
```
paste `./bench_qwen36b.sh` output
```
