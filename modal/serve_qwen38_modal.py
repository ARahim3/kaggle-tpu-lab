"""Serve Qwen3.8-27B GPTQ-Int4 with vLLM on Modal.

    modal deploy modal/serve_qwen38_modal.py                 # L40S, full 262K ctx
    BURST_GPU=A10G BURST_MAX_MODEL_LEN=6144 BURST_GPU_MEM_UTIL=0.98 \
    BURST_MAX_NUM_SEQS=4 BURST_MAX_NUM_BATCHED_TOKENS=1024 BURST_ENFORCE_EAGER=1 \
    modal deploy modal/serve_qwen38_modal.py                 # A10G free-credit profile

The API key comes from the qwen38-serve-key secret (BURST_SERVE_KEY) — an
unauthenticated public URL is a different, worse system. The container scales
to zero after BURST_SCALEDOWN_SECONDS idle; `modal app stop burst-qwen38`
ends the deployment for good.
"""
import logging
import os

import modal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VLLM_VERSION = "0.28.0"
APP_NAME = "burst-qwen38"
SERVE_PORT = 8000
REGION = "us-east"

# Deploy-time profile: one script, any supported GPU shape.
GPU = os.environ.get("BURST_GPU", "L40S")
MODEL_DIR = os.environ.get("BURST_WEIGHTS_DIR", "/weights/qwen38-27b-gptq-int4")
SCALEDOWN_SECONDS = int(os.environ.get("BURST_SCALEDOWN_SECONDS", "600"))


def _env_knob(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


image = (
    # The official vLLM image: vLLM 0.28 JIT-compiles flashinfer kernels for
    # this model's hybrid (GDN) attention on first boot and needs nvcc —
    # present here, absent from debian_slim (measured: two boots died there).
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}", add_python="3.12")
    # The image's own entrypoint swallows Modal's container arguments
    # (`vllm: error: unrecognized arguments: ...` retry loop).
    .entrypoint([])
    # JIT caches ride a Volume: flashinfer/triton artifacts would otherwise
    # rebuild on every cold boot (~2 min). The first boot pays the compile
    # and the Volume commits on container exit.
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "FLASHINFER_CACHE_DIR": "/jitcache/flashinfer",
            "TRITON_CACHE_DIR": "/jitcache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/jitcache/torchinductor",
            "BURST_WEIGHTS_DIR": MODEL_DIR,
        }
    )
)

with image.imports():
    import subprocess
    import time

volume = modal.Volume.from_name(
    os.environ.get("BURST_WEIGHTS_VOLUME", "burst-qwen38-27b-gptq-int4")
)
jit_cache = modal.Volume.from_name("burst-qwen38-jit-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _vllm_command() -> list:
    max_model_len = _env_knob("BURST_MAX_MODEL_LEN", "262144")
    mem_util = _env_knob("BURST_GPU_MEM_UTIL", "0.90")
    max_num_seqs = _env_knob("BURST_MAX_NUM_SEQS")
    max_batched_tokens = _env_knob("BURST_MAX_NUM_BATCHED_TOKENS")
    spec_config = _env_knob("BURST_SPECULATIVE_CONFIG")
    command = [
        "vllm", "serve",
        os.environ.get("BURST_WEIGHTS_DIR", MODEL_DIR),
        "--served-model-name", _env_knob("BURST_MODEL", "qwen38-27b"),
        "--api-key", os.environ["BURST_SERVE_KEY"],
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", mem_util,
        "--port", str(SERVE_PORT),
    ]
    if max_num_seqs:
        command += ["--max-num-seqs", max_num_seqs]
    if max_batched_tokens:
        command += ["--max-num-batched-tokens", max_batched_tokens]
    if _env_knob("BURST_ENFORCE_EAGER") == "1":
        command.append("--enforce-eager")
    if spec_config:
        # e.g. '{"method":"mtp","num_speculative_tokens":1}'
        command += ["--speculative-config", spec_config]
    return command


def _local_url(path):
    return f"http://127.0.0.1:{SERVE_PORT}{path}"


def _auth_headers():
    return {"Authorization": "Bearer " + os.environ["BURST_SERVE_KEY"]}


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/weights": volume, "/jitcache": jit_cache},
    secrets=[modal.Secret.from_name("qwen38-serve-key")],
    scaledown_window=SCALEDOWN_SECONDS,
    max_containers=1,
    startup_timeout=30 * 60,
    port=SERVE_PORT,
    routing_region=REGION,
    name="serve",
    unauthenticated=True,  # vLLM's --api-key middleware is the auth layer
    target_concurrency=32,
)
@modal.web_server(port=SERVE_PORT, startup_timeout=30 * 60)
def serve():
    import subprocess

    subprocess.Popen(_vllm_command())


@app.local_entrypoint()
def main():
    log.info("deploy with: modal deploy modal/serve_qwen38_modal.py")
