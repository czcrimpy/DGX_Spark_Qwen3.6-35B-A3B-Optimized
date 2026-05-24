#!/usr/bin/env bash
#
# install.sh — automated build pipeline for DGX Spark Qwen3.6-35B-A3B optimized.
#
# Phases:
#   0. Download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound (~18 GB)
#   1. (optional, --hybrid) Download Qwen/Qwen3.6-35B-A3B-FP8 (~35 GB)
#   2. (optional, --hybrid) Build hybrid INT4+FP8 checkpoint (~15-20 min, +1.8% perf)
#   3. (optional, --hybrid) Add MTP weights back to hybrid checkpoint
#   4. Ensure vllm-sm121 base image exists (clones eugr/spark-vllm-docker if needed)
#   5. Build vllm-qwen36b-v2 final image (INT8 LM Head patch + hybrid dispatch)
#
# Flags:
#   --hybrid      Build hybrid INT4+FP8 checkpoint (adds +1.8% speed, ~40 GB extra disk, ~20 min)
#                 Skip this unless you want the last few tok/s — Phase 3 config
#                 (INT4 + MTP + INT8 LM Head) gives 98% of the total benefit.
#   --launch      After build, auto-launch the container.
#   --no-launch   Never launch. Useful for unattended runs.
#   --no-cache    Wipe existing vllm-qwen36b-v2 image and BuildKit cache, rebuild from scratch.
#   -h | --help   Print this help and exit.
#
# Sudo: this script never invokes sudo. If a prerequisite is missing it prints
# the exact command to run and exits non-zero.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_VLLM_DIR="${PROJECT_DIR}/spark-vllm-docker"
HYBRID_DIR="${HOME}/models/qwen36b-hybrid-int4fp8"
SPARK_VLLM_PIN="49d6d9fefd7cd05e63af8b28e4b514e9d30d249f"
TORCH_NIGHTLY_DATE="20260408"
TORCH_VERSION="2.12.0.dev${TORCH_NIGHTLY_DATE}+cu130"
TORCHVISION_VERSION="0.27.0.dev${TORCH_NIGHTLY_DATE}+cu130"
TORCHAUDIO_VERSION="2.11.0.dev${TORCH_NIGHTLY_DATE}+cu130"

# ── Flags ─────────────────────────────────────────────────────────────────────
BUILD_HYBRID=0
LAUNCH_MODE="prompt"   # prompt | yes | no
NO_CACHE=0

for arg in "$@"; do
    case "$arg" in
        --hybrid)     BUILD_HYBRID=1 ;;
        --launch)     LAUNCH_MODE="yes" ;;
        --no-launch)  LAUNCH_MODE="no" ;;
        --no-cache)   NO_CACHE=1 ;;
        -h|--help)
            sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown flag: $arg (use --help)" >&2; exit 2 ;;
    esac
done

# ── Pretty output ─────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[1;33m'
    C_BLU=$'\033[0;34m'; C_CYN=$'\033[0;36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_CYN=""; C_DIM=""; C_OFF=""
fi

log()  { echo "${C_BLU}[install]${C_OFF} $*"; }
note() { echo "${C_DIM}          $*${C_OFF}"; }
ok()   { echo "${C_GRN}[ ok ]${C_OFF}    $*"; }
warn() { echo "${C_YEL}[warn]${C_OFF}    $*"; }
err()  { echo "${C_RED}[err ]${C_OFF}    $*" >&2; }
abort(){ err "$1"; exit 1; }

STEP_NUM=0
step() {
    STEP_NUM=$((STEP_NUM + 1))
    echo
    log "${C_CYN}▶ [${STEP_NUM}] $1${C_OFF}"
    if [ -n "${2:-}" ]; then
        note "$2"
    fi
}

# ── Prerequisites ─────────────────────────────────────────────────────────────
step "Checking prerequisites"

missing=()
check() {
    local label="$1" cmd="$2" fix="$3"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  ${C_GRN}✓${C_OFF} ${label}"
    else
        echo "  ${C_RED}✗${C_OFF} ${label}   ${C_DIM}— missing${C_OFF}"
        missing+=("${label}"$'\t'"${fix}")
    fi
}

check "python3"              "command -v python3"           "sudo apt install -y python3"
check "python3-venv"         "python3 -c 'import venv'"     "sudo apt install -y python3-venv python3-pip"
check "git"                  "command -v git"               "sudo apt install -y git"
check "curl"                 "command -v curl"              "sudo apt install -y curl"
check "docker"               "command -v docker"            "https://docs.docker.com/engine/install/ubuntu/"
check "docker no-sudo"       "docker info"                  "sudo usermod -aG docker \$USER && newgrp docker"

# Disk check
need_gb=$([ "$BUILD_HYBRID" = "1" ] && echo 80 || echo 40)
free_gb=$(df -BG "${HOME}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
free_gb=${free_gb:-0}
if [ "$free_gb" -ge "$need_gb" ]; then
    echo "  ${C_GRN}✓${C_OFF} free disk ≥ ${need_gb} GB (have ${free_gb} GB)"
else
    warn "only ${free_gb} GB free in \$HOME, need ~${need_gb} GB for this config"
fi

if [ "${#missing[@]}" -gt 0 ]; then
    echo
    err "${#missing[@]} prerequisite(s) missing:"
    for item in "${missing[@]}"; do
        what="${item%%$'\t'*}"; fix="${item#*$'\t'}"
        echo "  ${C_YEL}•${C_OFF} ${what}"
        echo "    ${C_CYN}${fix}${C_OFF}"
    done
    exit 1
fi

# ── Python venv + host deps ──────────────────────────────────────────────────
step "Python venv + host-side dependencies"

cd "${PROJECT_DIR}"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q torch numpy safetensors huggingface_hub

# ── Phase 0: download Intel INT4 ──────────────────────────────────────────────
step "Phase 0 — Downloading Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound" \
     "~18 GB, first time may take 10-20 min; cached: instant"

hf download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound
INTEL_DIR=$(hf download Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound --quiet)
[ -d "$INTEL_DIR" ] || abort "INTEL_DIR not found: ${INTEL_DIR}"
note "INTEL_DIR=${INTEL_DIR}"

# ── Phase 1+2+3: hybrid checkpoint (optional) ────────────────────────────────
if [ "$BUILD_HYBRID" = "1" ]; then
    step "Phase 1 — Downloading Qwen/Qwen3.6-35B-A3B-FP8" \
         "~35 GB, needed only for hybrid INT4+FP8 build"
    hf download Qwen/Qwen3.6-35B-A3B-FP8 >/dev/null

    # Check if hybrid checkpoint already exists
    if [ -f "${HYBRID_DIR}/model_extra_tensors.safetensors" ] \
        && [ -f "${HYBRID_DIR}/model.safetensors.index.json" ]; then
        step "Phase 2+3 — Hybrid checkpoint already exists, skipping"
        note "existing: ${HYBRID_DIR}"
    else
        step "Phase 2 — Building hybrid INT4+FP8 checkpoint" \
             "~15-20 min, output ~21 GB at ${HYBRID_DIR}"
        python3 "${PROJECT_DIR}/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" \
            --gptq-dir "${INTEL_DIR}" \
            --fp8-repo Qwen/Qwen3.6-35B-A3B-FP8 \
            --output "${HYBRID_DIR}" \
            --force

        step "Phase 3 — Adding MTP weights to hybrid checkpoint" \
             "restores 2329 MTP tensors that the hybrid build strips; critical for speculative decoding"
        python3 "${PROJECT_DIR}/patches/02-mtp-speculative/add-mtp-weights.py" \
            --source "${INTEL_DIR}" \
            --target "${HYBRID_DIR}"
    fi

    MODEL_SERVE_PATH="/models/qwen36b-hybrid-int4fp8"
    MODEL_MOUNT_SRC="$(dirname "${HYBRID_DIR}")"
else
    note "skipping hybrid build (use --hybrid to enable, adds ~2% speed)"
    MODEL_SERVE_PATH="Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound"
    MODEL_MOUNT_SRC=""  # no custom mount needed, HF cache is enough
fi

# ── --no-cache: wipe image and BuildKit cache ────────────────────────────────
if [ "$NO_CACHE" = "1" ]; then
    log "${C_YEL}--no-cache: removing existing image and pruning BuildKit${C_OFF}"
    docker rmi -f vllm-qwen36b-v2:latest 2>/dev/null || true
    docker builder prune -af >/dev/null 2>&1 || true
fi

# ── Phase 4: vllm-sm121 base image ────────────────────────────────────────────
if docker image inspect vllm-sm121:latest >/dev/null 2>&1; then
    step "Phase 4 — vllm-sm121 base image already exists, skipping"
    note "delete with 'docker rmi vllm-sm121' to rebuild, or pass --no-cache"
else
    step "Phase 4 — Building vllm-sm121 base image for SM121" \
         "first build: ~30-60 min; cached: ~3 min"

    if [ ! -d "${SPARK_VLLM_DIR}/.git" ]; then
        note "cloning eugr/spark-vllm-docker into ${SPARK_VLLM_DIR}"
        git clone https://github.com/eugr/spark-vllm-docker.git "${SPARK_VLLM_DIR}"
    else
        note "spark-vllm-docker already cloned, refreshing"
        git -C "${SPARK_VLLM_DIR}" fetch --quiet origin
    fi

    git -C "${SPARK_VLLM_DIR}" -c advice.detachedHead=false checkout --force "${SPARK_VLLM_PIN}"

    # Strip temporary PR patch blocks (see albond's install.sh for rationale)
    sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"
    sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"

    # Pin PyTorch nightly in both stages for ABI consistency
    sed -i "s|uv pip install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/cu130|uv pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} triton --index-url https://download.pytorch.org/whl/nightly/cu130|g" \
        "${SPARK_VLLM_DIR}/Dockerfile"

    # Let uv wait longer on NVIDIA wheels (pypi.nvidia.com can be slow/flaky on aarch64)
    uv_timeout_block=$'ENV UV_HTTP_TIMEOUT=600\nENV UV_HTTP_RETRIES=10'
    if ! grep -q '^ENV UV_HTTP_TIMEOUT=' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i "/^RUN --mount=type=cache,id=uv-cache,target=\\/root\\/\.cache\\/uv/i ${uv_timeout_block}" \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    # Suppress CUTLASS×CUDA13 deprecation spam
    if ! grep -q 'NVCC_APPEND_FLAGS' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i '/^ENV TORCH_CUDA_ARCH_LIST=/a ENV NVCC_APPEND_FLAGS="-Xcompiler=-Wno-deprecated-declarations -diag-suppress=20012 -diag-suppress=20013 -diag-suppress=20014 -diag-suppress=20015"' \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    (
        cd "${SPARK_VLLM_DIR}"
        ./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5
    )
    docker tag vllm-sm121:latest vllm-node-tf5:latest

    docker image inspect vllm-sm121:latest >/dev/null 2>&1 \
        || abort "vllm-sm121:latest not built"
fi

# ── Phase 5: build final image ────────────────────────────────────────────────
if docker image inspect vllm-qwen36b-v2:latest >/dev/null 2>&1 && [ "$NO_CACHE" = "0" ]; then
    step "Phase 5 — vllm-qwen36b-v2 image already exists, skipping"
    note "delete with 'docker rmi vllm-qwen36b-v2' to rebuild, or pass --no-cache"
else
    step "Phase 5 — Building vllm-qwen36b-v2 final image" \
         "thin layer over vllm-sm121: INT8 LM Head patch + hybrid INT4+FP8 dispatch"
    cd "${PROJECT_DIR}"
    docker build -t vllm-qwen36b-v2 -f docker/Dockerfile.v2 .
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo
ok "${C_GRN}All build steps complete${C_OFF}"
echo
log "Images:"
docker images vllm-sm121     --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
docker images vllm-qwen36b-v2 --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
echo

# ── Launch (prompt or auto) ──────────────────────────────────────────────────
CHAT_TEMPLATE_SRC="${PROJECT_DIR}/configs/chat_template.jinja"

build_launch_cmd() {
    local mount_model_arg=""
    if [ -n "${MODEL_MOUNT_SRC}" ]; then
        mount_model_arg="-v ${MODEL_MOUNT_SRC}:/models"
    fi

    cat <<EOF
docker run -d --name vllm-qwen36b \\
    --gpus all --net=host --ipc=host \\
    -v \${HOME}/.cache/huggingface:/root/.cache/huggingface \\
    -v ${CHAT_TEMPLATE_SRC}:/opt/unsloth.jinja:ro \\
    ${mount_model_arg} \\
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \\
    vllm-qwen36b-v2 \\
    serve ${MODEL_SERVE_PATH} \\
    --served-model-name qwen --port 8000 --host 0.0.0.0 \\
    --max-model-len 262144 --max-num-batched-tokens 16384 \\
    --gpu-memory-utilization 0.90 \\
    --reasoning-parser qwen3 \\
    --attention-backend FLASHINFER \\
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \\
    --load-format fastsafetensors --trust-remote-code \\
    --chat-template /opt/unsloth.jinja \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \\
    -tp 1
EOF
}

print_launch() {
    cat <<EOF
${C_CYN}To launch manually:${C_OFF}

$(build_launch_cmd)

Wait ~3-4 min for model load + warmup, then:
    curl http://127.0.0.1:8000/health
    curl http://127.0.0.1:8000/v1/models
EOF
}

do_launch() {
    log "Launching vllm-qwen36b..."
    if docker ps -a --format '{{.Names}}' | grep -qx vllm-qwen36b; then
        warn "container 'vllm-qwen36b' exists — removing"
        docker rm -f vllm-qwen36b >/dev/null
    fi
    eval "$(build_launch_cmd)" || abort "docker run failed"
    ok "container started. Use 'docker logs -f vllm-qwen36b' to watch startup."
    note "endpoint will be: http://127.0.0.1:8000/v1"
    note "stop: docker stop vllm-qwen36b"
    note "benchmark: ./bench_qwen36b.sh v2"
}

case "$LAUNCH_MODE" in
    yes)
        echo; do_launch ;;
    no)
        echo; print_launch ;;
    prompt)
        echo
        if [ ! -t 0 ]; then
            note "non-interactive shell — skipping launch prompt"
            print_launch
        else
            read -r -p "${C_CYN}Launch the container now? [y/N] ${C_OFF}" reply
            if [[ "${reply}" =~ ^[Yy]$ ]]; then
                do_launch
            else
                print_launch
            fi
        fi
        ;;
esac
