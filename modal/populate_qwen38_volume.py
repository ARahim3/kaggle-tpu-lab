"""Populate the qwen38-27b GPTQ-Int4 Modal Volume (one-time, ~19 GiB).

    python modal/populate_qwen38_volume.py

Downloads the quantized checkpoint into the Volume so the serving app
boots offline (HF_HUB_OFFLINE=1) — no per-boot download cost.
"""
import logging

import modal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VOLUME_NAME = "burst-qwen38-27b-gptq-int4"
REPO_ID = "Qwen/Qwen3.8-27B-GPTQ-Int4"
MODEL_DIR = "/vol/qwen38-27b-gptq-int4"

image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]")

app = modal.App("burst-qwen38-populate", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(volumes={"/vol": volume}, timeout=60 * 60)
def populate():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        max_workers=8,
    )
    volume.commit()
    log.info("populated %s from %s", VOLUME_NAME, REPO_ID)


@app.local_entrypoint()
def main():
    populate.remote()
