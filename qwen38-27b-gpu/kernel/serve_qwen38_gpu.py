"""
Serve Qwen3.8-27B (AWQ-INT4) on a Kaggle GPU session (Tesla T4 x2) with
stock vLLM — the GPU leaf of the Kaggle substrate.

TPU quota (20 h/week) starves before GPU quota (30 h/week) is touched at
all, and the T4 shape is the only GPU Kaggle grants a script kernel that
the burst control plane declares (config.rs: `NvidiaTeslaT4`). 32 GB
across two cards cannot hold the 54 GB bf16 checkpoint, so this probe
serves the community AWQ-INT4 quant (~16 GB) with tensor parallelism 2.
AWQ dequant kernels run on SM75 (Turing); flashinfer/marlin paths need
SM80+, so this is stock vLLM — sglang/DFlash2 are out of scope here.

Purpose (docs/KAGGLE-GPU-COLDSTART.md): measure the GPU cold-start stage
breakdown with the same `BURST_STAGE` instrumentation the TPU kernel
uses, serve bench cells through a cloudflared tunnel, and harvest the
vLLM torch.compile cache + a manifest as session output so a follow-up
run can attach them as a dataset (Kaggle's persistence is datasets, not
snapshots).

Steps (each announced in the log):
  1/6  runtime  — venv with vllm built by uv (~2-4 min: CUDA torch wheels)
  2/6  weights  — find a mounted weights dataset, else HF-download the AWQ;
                  also restore the compile cache from its dataset when one is
                  attached (G2), beside the install
  3/6  server   — start vLLM (TP=2, text-only)
  4/6  tunnel   — open a public cloudflared URL
  5/6  ready    — READY banner + self-test, then keep serving until
                  keepalive_min elapses or idle_min passes with no request;
                  on the exit path, harvest the torch.compile cache
"""
import collections
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# The AWQ pack sits ~150 MB over the T4s' headroom at create_weights
# (sessions 3-5, 2026-10-14); allocator fragmentation is that large.
# expandable_segments removes the cudaMalloc-fragmentation tax; the server
# subprocess inherits this (launch_server copies os.environ).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line; the notebook drops serve_config.json beside it)
# History: the sessions 3-6 "OOM at create_weights" was NOT a memory wall:
# b838c97 double-launched the engine (once before wait_healthy, once inside
# it), so two full engines loaded concurrently — two vllm_dist init groups,
# ~7 GiB held per process per card. Session 7 (2026-10-14) is the first
# single-engine run at the 16k/4/0.88 shape; the old CFG literal that pinned
# a different shape here is now the DEFAULTS block below.

DEFAULTS = {
    "vllm_version": "0.29.0",      # pinned to the version the G1 harvest manifest recorded —
                                   # a compile cache is only valid for the stack that built it
    "vllm_cache_dataset": "",      # owner/slug of a mounted compile-cache dataset; "" = cold
                                   # compile. G2: preload VLLM_CACHE_ROOT from the G1 harvest
                                   # (michealkinney/qwen38-vllm-cache-t4) so torch.compile loads
                                   # instead of recompiling — the 665 s engine-init question
    "hf_model_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",  # ~16 GB AWQ-INT4, fits T4x2 with TP=2
    "weights_dataset": "",         # owner/slug of a mounted AWQ weights dataset; "" = HF download.
                                   # Attaching one skips a ~16 GB read — the GPU analog of the
                                   # TPU weights dataset and the biggest cold-start lever
    "max_model_len": 16384,        # overridden by the CFG literal above — kept in sync as the
                                   # default; 32k x 8 seqs OOM'd both T4s (measured 2026-10-14,
                                   # sessions 3-4: 14.56 GiB/card usable, ~9.8 GiB pack/card
                                   # with TP=2). 16k x 4 seqs = ~2 GiB KV/card + headroom.
    "max_num_seqs": 4,
    "gpu_mem_util": 0.88,
    "enforce_eager": False,        # False measures real CUDA-graph capture cost; True trades
                                   # decode speed for a shorter startup — a cold-start axis
    "attention_backend": "",       # "" = vLLM auto (Turing falls back to XFORMERS); override
                                   # e.g. "XFORMERS" only to pin a broken auto-pick
    "kv_cache_dtype": "auto",      # "auto" (bf16) | "fp8" — fp8 KV is an SM75 question mark,
                                   # exactly what a probe is for
    "tool_call_parser": "qwen3_coder",
    "reasoning_effort_default": "xhigh",
    "keepalive_min": 120,          # Kaggle GPU sessions cap at 12 h; the probe wants hours, not half a day
    "idle_min": 10,
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",
    "served_model_name": "qwen38-27b-awq",
    "verbose": False,
    "harvest": True,               # tar the torch.compile cache + write manifest into /kaggle/working
    # burst control-plane reporting (same contract as serve_qwen38.py)
    "burst_gateway": "",
    "burst_instance": "",
    "burst_gateway_token": "",
    "burst_credential_ref": "session/qwen38",
    "burst_model": "qwen38-27b-awq",
}
CFG = {**DEFAULTS, **(CFG or {})}
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)


def report_to_gateway(reason):
    """Announce {instance, url, credential_ref, model} to the burst gateway.

    Non-fatal by design; the gateway token is sent as the bearer, never logged.
    """
    if not (CFG["burst_gateway"] and url and CFG["burst_instance"]):
        return False
    body = {
        "instance": CFG["burst_instance"],
        "url": f"{url}/v1",
        "credential_ref": CFG["burst_credential_ref"] or None,
        "model": CFG["burst_model"] or None,
    }
    headers = {"Content-Type": "application/json",
               "User-Agent": "burst-kernel/0.1"}
    if CFG["burst_gateway_token"]:
        headers["Authorization"] = f"Bearer {CFG['burst_gateway_token']}"
    req = urllib.request.Request(
        CFG["burst_gateway"].rstrip("/") + "/admin/endpoints",
        data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            log(f"   reported endpoint to gateway ({reason}): HTTP {r.status}")
            return r.status in (200, 201, 202)
    except Exception as e:
        log(f"   gateway report failed ({reason}): {e}")
        return False


PORT = 8000
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
VLLM_CACHE = "/tmp/vllm_cache"     # VLLM_CACHE_ROOT: torch.compile artifacts (harvestable)
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"
STAGES = {}                        # stage -> seconds, for the harvested manifest

os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["VLLM_CACHE_ROOT"] = VLLM_CACHE
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor"

_raw = open(RAW_LOG, "a", buffering=1)


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: one parseable `BURST_STAGE` line (same contract as the
    TPU kernel) plus the human PHASE line, and ntfy when a topic is set."""
    event = {"stage": phase, "t_ms": int((time.time() - T0) * 1000)}
    if "secs" in extra:
        event["dur_ms"] = int(float(extra["secs"]) * 1000)
        STAGES[phase] = round(float(extra["secs"]), 1)
    event.update(extra)
    log("BURST_STAGE", json.dumps(event))
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"burst-gpu {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    """Datasets mount at /kaggle/input/<slug> (UI) or /kaggle/input/datasets/<owner>/<slug> (API push)."""
    for pat in patterns:
        hits = glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
        if hits:
            return hits[0]
    return None


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return
    try:
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


def gpu_inventory():
    """Log the granted GPUs and their compute capability — the first thing a
    failed kernel run should tell us (SM75 gates which kernels exist)."""
    caps = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        for ln in r.stdout.strip().splitlines():
            log(f"   gpu: {ln.strip()}")
    except Exception as e:
        log(f"   nvidia-smi failed: {e}")
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            caps.append(torch.cuda.get_device_capability(i))
        log(f"   compute capabilities: {caps}")
        return {"capabilities": [list(c) for c in caps]}
    except Exception as e:
        log(f"   torch capability probe failed: {e}")
        return {}


def install_runtime():
    """Fresh venv with stock vLLM via uv. CUDA wheels are ~2-3 GB — the price
    of the GPU path the TPU kernel avoided with cpu torch."""
    log("   building venv with uv (this downloads the CUDA torch stack)...")
    # wrapt: vLLM's server path imports it on this stack without declaring it
    # (measured 2026-10-14: ModuleNotFoundError crash-loop at engine init).
    # Requirements are separate argv entries — uv parses one string with a
    # space as a single malformed requirement ("Trailing `wrapt` is not
    # allowed", measured same day).
    reqs = [f"vllm=={CFG['vllm_version']}" if CFG["vllm_version"] else "vllm", "wrapt"]
    if (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip") == 0
            and sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"], "uv") == 0):
        if sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
               "--torch-backend=auto", *reqs], "uv") == 0:
            return "uv"
        log("   --torch-backend=auto failed; retrying with plain resolution")
        if sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY, *reqs], "uv") == 0:
            return "uv"
    log("   uv failed; falling back to pip (slower)")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh([sys.executable, "-m", "venv", VENV], "venv") != 0:
        return None
    if sh([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"], "pip") != 0:
        return None
    return "pip" if sh([PY, "-m", "pip", "install", "-q", *reqs], "pip") == 0 else None


def runtime_ok():
    r = subprocess.run([PY, "-c", "import torch, vllm; print(torch.__version__, vllm.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        log(f"   runtime check OK (torch/vllm {r.stdout.strip()})")
        return True
    log("   runtime check FAILED:", (r.stderr or r.stdout)[-800:])
    return False


def resolve_weights():
    """Mount a weights dataset when one is attached, else HF-download the AWQ."""
    if CFG["weights_dataset"]:
        path = find_input(CFG["weights_dataset"].split("/")[-1])
        if path and os.path.exists(os.path.join(path, "config.json")):
            publish("weights-mounted", path=path)
            return path
        log(f"   weights dataset {CFG['weights_dataset']} not mounted -> HF download")
    publish("weights-download", model=CFG["hf_model_id"],
            note="attach a weights dataset to skip a ~16 GB read")
    t = time.time()
    from huggingface_hub import snapshot_download
    path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges*"])
    secs = int(time.time() - t)
    publish("weights-downloaded", secs=secs)
    size = sum(os.path.getsize(f) for f in glob.glob(path + "/**/*", recursive=True)
               if os.path.isfile(f))
    log(f"   {size / 2**30:.1f} GiB in {secs} s ({size / max(secs, 1) / 2**20:.0f} MiB/s)")
    return path


def _find_cache_artifact(ds):
    """Locate the cache artifact under the mounted dataset dir.

    The published dataset (qwen38-vllm-cache-t4 v2) carries the harvest as a
    LOOSE TREE: vllm_cache/vllm_cache/torch_compile_cache/... + manifest.json.
    A tar of the same tree is accepted as the preferred shape but is not what
    is published today. Returns ("tar", path) or ("tree", cache_root) or None.
    """
    tar = os.path.join(ds, "vllm_cache.tar")
    if os.path.exists(tar):
        return "tar", tar
    for root, dirs, _files in os.walk(ds):
        if "torch_compile_cache" in dirs:
            return "tree", root     # dir whose CONTENTS are the VLLM_CACHE_ROOT
        if root.rstrip("/").count("/") - ds.rstrip("/").count("/") >= 4:
            dirs[:] = []            # bound the walk: cache root sits near the top
    return None


def restore_compile_cache():
    """Preload VLLM_CACHE_ROOT from the attached cache dataset (G2).

    Accepts either artifact shape: a vllm_cache.tar (extracted to /tmp) or the
    published loose tree (copied file-by-file). Runs beside the venv install —
    the mount exists from session start. Kaggle materializes /kaggle/input
    lazily (v2/v3 broke on variants of this), so the poll keys on the artifact
    itself, not the mount dir. A version mismatch degrades to an honest cold
    start (the cache is content-addressed, so it can miss but not corrupt),
    and check_cache_versions says so loudly.
    """
    if not CFG["vllm_cache_dataset"]:
        publish("cache-restore", note="no cache dataset attached (cold compile)")
        return
    slug = CFG["vllm_cache_dataset"].split("/")[-1]
    artifact, waited = None, 0.0
    while waited <= 240:
        ds = find_input(slug)
        if ds and os.path.isdir(ds):
            artifact = _find_cache_artifact(ds)
            if artifact:
                break
        time.sleep(5)
        waited += 5
    if not artifact:
        publish("cache-restore", ok=False,
                note=f"dataset {CFG['vllm_cache_dataset']} carried no tar or torch_compile_cache tree after {int(waited)} s")
        return
    kind, path = artifact
    t = time.time()
    if kind == "tar":
        rc = sh(["tar", "-xf", path, "-C", "/tmp"], "cache")
    else:
        os.makedirs(VLLM_CACHE, exist_ok=True)
        rc = sh(["cp", "-a", os.path.join(path, "."), VLLM_CACHE], "cache")
    secs = time.time() - t
    if rc != 0:
        publish("cache-restore", ok=False, note=f"{kind} restore failed; falling back to cold compile")
        return
    subprocess.run(["chmod", "-R", "u+w", VLLM_CACHE], capture_output=True)
    files = glob.glob(VLLM_CACHE + "/**/*", recursive=True)
    n = sum(1 for f in files if os.path.isfile(f))
    gib = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 2**30
    publish("cache-restored", secs=round(secs, 1), entries=n, cache_gib=round(gib, 2), shape=kind)
    log(f"   restored {n} cache entries ({gib:.2f} GiB) from {kind} in {secs:.0f} s -> {VLLM_CACHE}")


def check_cache_versions():
    """Loud guard for the G2 decision rule: the restored cache only counts if
    the installed stack matches the manifest's recorded versions."""
    if not CFG["vllm_cache_dataset"]:
        return
    slug = CFG["vllm_cache_dataset"].split("/")[-1]
    ds = find_input(slug)
    mf_path = os.path.join(ds, "manifest.json") if ds else None
    if not mf_path or not os.path.exists(mf_path):
        return
    try:
        built = json.loads(Path(mf_path).read_text()).get("versions", {})
    except Exception:
        return
    r = subprocess.run([PY, "-c",
                        "import torch, vllm, sys; print(torch.__version__, vllm.__version__, sys.version.split()[0])"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return
    torch_v, vllm_v, py_v = r.stdout.split()
    mismatches = [f"{k}: cache={built.get(k)} runtime={cur}" for k, cur in
                  (("vllm", vllm_v), ("torch", torch_v), ("python", py_v))
                  if built.get(k) and built.get(k) != cur]
    if mismatches:
        log("   !! CACHE VERSION MISMATCH — the cache may miss; expect cold-start compile")
        for m in mismatches:
            log(f"      {m}")
    else:
        log(f"   cache manifest versions match the runtime (vllm {vllm_v}, torch {torch_v})")


NOISE = ("Triton is installed", "Transparent hugepages", "Pin memory", "Expect torch.Tensor",
         "Inductor compilation", "SyntaxWarning", "torch_dtype", "UserWarning",
         "warnings.warn", "resource_tracker", "it is strongly recommended",
         " FlashAttention", "not compatible with")


def make_translator():
    """Turns vLLM's firehose into a handful of human lines."""
    st = {"loads": 0, "said": set()}

    def once(key, msg):
        if key not in st["said"]:
            st["said"].add(key)
            log(msg)

    def tr(line):
        if CFG["verbose"]:
            print(f"[vllm] {line[:500]}", flush=True)
            return
        if any(k in line for k in NOISE):
            return
        m = re.search(r"Loading weights took ([\d.]+) seconds", line)
        if m:
            st["loads"] += 1
            if st["loads"] == 1:
                log(f"   weights read in {float(m.group(1)):.0f} s")
            return
        m = re.search(r"# GPU blocks: ([\d,]+)", line)
        if m:
            log(f"   KV cache: {m.group(1)} GPU blocks "
                f"({int(m.group(1).replace(',', '')) * 16 // 1024}k tokens)")
            return
        if "Capturing CUDA graphs" in line or "Capture cuda graph" in line:
            once("graphs", "   capturing CUDA graphs (~tens of seconds; enforce_eager skips this)")
            return
        if "Compilation" in line or "torch.compile" in line:
            once("compile", "   torch.compile — cached on the second boot (VLLM_CACHE_ROOT harvest)")
            return
        if "Application startup complete" in line:
            return
        if " ERROR " in line or "Traceback" in line or "Error:" in line or "rror(" in line:
            print(time.strftime("[%H:%M:%S] ") + f"   [vllm] {line[:400]}", flush=True)
    return tr


def server_args(cfg, model_path):
    args = [PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", "2",
            "--max-model-len", str(cfg["max_model_len"]),
            "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--gpu-memory-utilization", str(cfg["gpu_mem_util"]),
            "--kv-cache-dtype", cfg["kv_cache_dtype"],
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            "--reasoning-parser", "qwen3",
            # Qwen3.8 is a vision-language checkpoint; the probe serves text.
            "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["enforce_eager"]:
        args += ["--enforce-eager"]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["reasoning_effort_default"] != "xhigh":
        tc = json.loads(Path(model_path, "tokenizer_config.json").read_text())
        template = tc["chat_template"].replace(
            "reasoning_effort|default('xhigh')",
            f"reasoning_effort|default('{cfg['reasoning_effort_default']}')")
        Path("/tmp/chat_template.jinja").write_text(template)
        args += ["--chat-template", "/tmp/chat_template.jinja"]
    return args


def launch_server(cfg, model_path):
    publish("server-launch", max_model_len=cfg["max_model_len"],
            max_num_seqs=cfg["max_num_seqs"], enforce_eager=cfg["enforce_eager"],
            kv_cache_dtype=cfg["kv_cache_dtype"])
    tail = collections.deque(maxlen=200)
    p = subprocess.Popen(server_args(cfg, model_path), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    tr = make_translator()

    def pump():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                _raw.write(f"[vllm] {line}\n")
                tr(line)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def healthy(cfg):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def server_busy():
    """True while any request is running or queued. Returns True on blindness
    so a monitoring failure can never idle-kill a serving session."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:
        return True
    for gauge in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        for line in text.splitlines():
            if line.startswith(gauge):
                try:
                    if float(line.rsplit(None, 1)[1]) > 0:
                        return True
                except (IndexError, ValueError):
                    pass
                break
    return False


def wait_healthy(cfg, model_path, max_heal_rounds=3):
    """Wait for /v1/models to answer. If the server dies on an undeclared
    import (vLLM's metadata has gaps on some stacks), install the named
    module into the venv and relaunch — bounded, so a genuinely broken
    dependency still fails the cell loudly instead of looping forever."""
    healed = set()
    rounds = 0
    server = launch_server(cfg, model_path)
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            tail = list(server.tail)
            missing = [m for ln in tail
                       for m in re.findall(r"No module named '([A-Za-z0-9_]+)'", ln)
                       if m not in healed]
            if missing and rounds < max_heal_rounds:
                rounds += 1
                for m in missing:
                    healed.add(m)
                    log(f"   server died on missing module '{m}' — installing "
                        f"and relaunching (heal round {rounds}/{max_heal_rounds})")
                    if sh([sys.executable, "-m", "uv", "pip", "install",
                           "--python", PY, m], f"heal-{m}") != 0:
                        sh([PY, "-m", "pip", "install", "-q", m], f"heal-{m}-pip")
                server = launch_server(cfg, model_path)
                continue
            log(f"server exited rc={server.returncode}; last output:\n"
                + "\n".join(tail[-100:]))
            publish("failed", step="server", rc=server.returncode,
                    tail="\n".join(tail)[-2500:])
            sys.exit(1)
        if healthy(cfg):
            if rounds:
                log(f"   healthy after {rounds} heal round(s): {sorted(healed)}")
            # Returns (proc, startup_secs): the caller owns the proc for
            # poll/shutdown in the keepalive loop. Session-7 lesson — when
            # the launch moved inside this function (f3aee35), returning
            # only seconds orphaned the `server` name and the notebook died
            # on `shutdown(server)` right after a green self-test.
            return server, int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("starting", elapsed_s=el)
            log(f"   ... {el // 60} min into startup")
        time.sleep(5)
    publish("failed", step="health-timeout",
            tail="\n".join(list(server.tail)[-60:])[-2500:])
    sys.exit(1)


def completion(cfg, prompt, max_tokens, stream=False, timeout=900):
    body = {"model": cfg["served_model_name"], "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
            if obj.get("usage"):
                gen = obj["usage"].get("completion_tokens", gen)
    return ttft, time.time() - t0, gen


def self_test(cfg):
    """Warm the request path and report single-stream decode speed."""
    tps = None
    try:
        completion(cfg, "Hello", 8)
        txt = completion(cfg, "The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion(cfg, "Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
    return tps


def harvest():
    """Tar the torch.compile cache and write a manifest as session output.

    Kaggle's persistence is datasets, not snapshots: whatever should survive
    the session has to land in /kaggle/working and be published as a dataset.
    The compile cache is the GPU analog of the TPU XLA cache harvest — the
    artifact a second boot needs to skip recompilation.
    """
    if not CFG["harvest"]:
        return
    banner(6, "harvest", f"{VLLM_CACHE} -> {WORK}")
    files = glob.glob(VLLM_CACHE + "/**/*", recursive=True)
    n = sum(1 for f in files if os.path.isfile(f))
    gib = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 2**30
    versions = {}
    try:
        r = subprocess.run([PY, "-c",
                            "import torch, vllm, sys; print(torch.__version__, vllm.__version__, sys.version.split()[0])"],
                           capture_output=True, text=True)
        torch_v, vllm_v, py_v = r.stdout.split()
        versions = {"torch": torch_v, "vllm": vllm_v, "python": py_v}
    except Exception:
        versions = {"python": PY_VER}
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "accelerator": "Kaggle GPU T4 x2",
        "model": CFG["hf_model_id"],
        "config": {k: CFG[k] for k in
                   ("max_model_len", "max_num_seqs", "gpu_mem_util",
                    "enforce_eager", "kv_cache_dtype")},
        "stages_secs": STAGES,
        "compile_cache": {"entries": n, "gib": round(gib, 3)},
        "versions": versions,
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if n:
        local = Path("/tmp/vllm_cache.harvest.tar")
        if sh(["tar", "-cf", str(local), "-C", os.path.dirname(VLLM_CACHE),
               os.path.basename(VLLM_CACHE)], "harvest") == 0:
            shutil.copy(local, WORK / "vllm_cache.tar")
            local.unlink(missing_ok=True)
            log(f"   harvested {n} entries ({gib:.2f} GiB) -> {WORK}/vllm_cache.tar")
    log(f"   manifest -> {WORK}/manifest.json")
    publish("cache-harvested", entries=n, cache_gib=round(gib, 2))


def shutdown(server):
    server.terminate()
    try:
        server.wait(timeout=120)
    except Exception:
        pass
    harvest()
    sys.exit(0)


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"stock vLLM {'pinned ' + CFG['vllm_version'] if CFG['vllm_version'] else '(latest)'}")
threading.Thread(target=fetch_cloudflared, daemon=True).start()
gpu_inventory()

# ---------------- 2. weights + cache (in parallel with the install) ----------------
weights = {}


def prepare_weights():
    try:
        weights["path"] = resolve_weights()
    except Exception as e:
        log(f"   weight resolution failed: {e}")
        weights["path"] = None


weight_loader = threading.Thread(target=prepare_weights, daemon=True)
weight_loader.start()


def prepare_cache():
    try:
        restore_compile_cache()
    except Exception as e:
        log(f"   cache restore failed: {e}")


cache_loader = threading.Thread(target=prepare_cache, daemon=True)
cache_loader.start()

t = time.time()
publish("install", vllm=CFG["vllm_version"] or "latest")
runtime = install_runtime()
if runtime is None or not runtime_ok():
    publish("failed", step="install")
    sys.exit(1)
publish("installed", secs=int(time.time() - t), via=runtime)

weight_loader.join()
cache_loader.join()
check_cache_versions()
model_path = weights.get("path")
if not model_path:
    publish("failed", step="weights")
    sys.exit(1)

if CFG["attention_backend"]:
    os.environ["VLLM_ATTENTION_BACKEND"] = CFG["attention_backend"]

# ---------------- 3. vLLM server ----------------
# Launched INSIDE wait_healthy() below, which owns the self-heal relaunch.
# b838c97..21c56ab regression note: this line used to launch the server AND
# wait_healthy launched it again — two full engines loaded concurrently
# (two vllm_dist init groups in the log) and every session 3-6 OOM'd both
# T4s at create_weights with ~7 GiB held per process per card. One launch,
# here absent on purpose.
banner(3, "Starting vLLM", f"TP=2 on T4x2, ctx {CFG['max_model_len']}, "
       f"{CFG['max_num_seqs']} seqs, eager={CFG['enforce_eager']}")

# ---------------- 4. tunnel (in parallel with the server start) ----------------
banner(4, "Public URL")
url = None
tunnel = None
for _ in range(60):
    if CLOUDFLARED.exists():
        break
    time.sleep(2)
if CLOUDFLARED.exists():
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                               "--no-autoupdate", "--protocol", "quic"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lines = []

    def pump_cf():
        for line in tunnel.stdout:
            lines.append(line.rstrip())
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_cf, daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline and url is None:
        for ln in lines:
            m = pat.search(ln)
            if m:
                url = m.group(0).rstrip("/")
                break
        time.sleep(1)
if url:
    log(f"   your endpoint will be  {url}/v1")
    log("   (not live yet — it answers 502 until the READY banner below)")
    publish("tunnel-url", endpoint=f"{url}/v1")
else:
    publish("tunnel-failed", note="server still reachable inside the kernel on :8000")

# ---------------- 5. wait, announce, self-test ----------------
server, startup = wait_healthy(CFG, model_path)
publish("serving", startup_secs=startup)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
        model=CFG["served_model_name"], startup_secs=startup)
report_to_gateway("ready")
self_test(CFG)

# ---------------- 6. keep alive, then harvest on the way out ----------------
t_serve = time.time()
idle_deadline = CFG["idle_min"] * 60
last_busy = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    if server.poll() is not None:
        publish("stopped", reason="server-exit", rc=server.returncode)
        sys.exit(1)
    busy = server_busy()
    if busy:
        last_busy = time.time()
    up = int((time.time() - t_serve) / 60)
    idle_min = int((time.time() - last_busy) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None),
                busy=busy, idle_min=idle_min)
        log(f"   still serving ({up} min, idle {idle_min}) — {url + '/v1' if url else ''}")
        report_to_gateway("heartbeat")
    if idle_deadline and (time.time() - last_busy) >= idle_deadline:
        publish("idle-shutdown", idle_min=idle_min,
                served_min=int((time.time() - t_serve) / 60))
        log(f"#  idle for {idle_min} min — exiting to free the GPU slot")
        shutdown(server)
publish("auto-shutdown", served_min=int((time.time() - t_serve) / 60))
shutdown(server)
