"""
Serve GLM-5.3-Flash (320B MoE, 18B active) on a Kaggle TPU v5e-8 with our own JAX engine.

This script is pushed to Kaggle as a script kernel by ../../launch.py, which fills in the CFG line below and embeds
the engine package. It also runs standalone (pasted into a Kaggle notebook next to a `glm53/` folder written by the
previous cell) — then it prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — pre-flight (datasets attached, Internet on, a real TPU: ~20 s, before anything slow), pinned libtpu
                  + a few pip packages (~1 min), cloudflared, the engine package
  2/6  weights  — the routed experts (3-bit codebook tables, Unsloth's UD-IQ3_XXS GGUF) and the non-expert weights
                  (int8) straight onto the eight chips (~6 min); the non-expert weights come from the serve dataset's
                  base store when it is attached, else from the FP8 checkpoint datasets
  3/6  vision   — the vision tower, sharded over the chips
  4/6  warm-up  — the prefill buckets, the batched decode and speculative-decoding programs, the snapshot programs:
                  loaded from the serve dataset's exported programs and compile cache; without them every program is
                  traced, exported and compiled (~28 min)
  5/6  tunnel   — a public cloudflared URL (three attempts; a URL that never resolves is replaced once)
  6/6  ready    — READY banner + self-test, then keep serving until keepalive_min elapses

A cold run leaves `jax_cache/` (everything it compiled), `exported/` (every program it traced) and, when it read the
FP8 checkpoint, `base/` (the non-expert weights, vision tower and tokenizer, ~11 GB) in /kaggle/working, so its output
can be turned into the serve dataset ("New dataset" from the kernel output) that later runs attach.
"""
import base64, collections, glob, hashlib, hmac, io, ipaddress, json, os, queue, re, secrets, shutil, socket, subprocess, sys, tarfile, threading, time, urllib.parse, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "expert_datasets": ["rahim3/glm53-flash-iq3xxs-1", "rahim3/glm53-flash-iq3xxs-2"],   # Unsloth UD-IQ3_XXS GGUF, mirrored
    "base_datasets": ["rahim3/glm53-flash-fp8-1", "rahim3/glm53-flash-fp8-2",            # HF FP8 checkpoint, mirrored:
                      "rahim3/glm53-flash-fp8-3", "rahim3/glm53-flash-fp8-4"],           # non-expert weights, vision, tokenizer
    "serve_dataset": "rahim3/glm53-flash-serve-v3",   # base/ (non-expert weights, vision, tokenizer) + exported/ (the
                                     # programs' exports) + jax_cache/ (their compiled executables); attached: the FP8
                                     # datasets are not needed and the warm-up neither traces nor compiles
    "serve_fallback": ["rahim3/glm53-flash-serve-v2", "rahim3/glm53-flash-serve"],   # used when `serve_dataset` is not
                                     # attached: an older serve dataset's base/ still fits; its compiled programs are
                                     # another build's (every program is traced and compiled, ~28 min)
    "aot": True,                     # AOT export of the programs (glm53.aot): a program that has to be traced is exported
                                     # to /kaggle/working/exported and a run that finds its file there, or in the serve
                                     # dataset's exported/, skips JAX's tracing and lowering (the compile cache only
                                     # skips XLA's compile); v6e 2026-10-06: warm-up 10.4 min with the cache -> 0.3
    "dump_base": True,               # a cold run writes base/ into its output (so the serve dataset can be made from it)
    "libtpu": "0.0.42.*",            # the image's runtime is 140x slower on gathers and cannot run Pallas kernels
    "max_len": 262144,               # context capacity (tokens); a multiple of 32
    "streams": 3,                    # requests decoded together (one program set per batch size, ~1-2.5 min compile
                                     # each); a 4th 262k set never passes the admission (`min_free_gb`) on the v5e budget
    "sets": 4,                       # cache sets on the chips: running streams + finished contexts kept for their next turn
    "piece": 1024,                   # prefill piece (tokens): 1024 = ~2,500-2,700 tok/s vs ~2,050-2,150 at 512 (v6e
    "sched_piece": 1024,             # 2026-10-04, 16k-128k prompts); three 262k streams either way since the admission
                                     # counts live buffers (`min_free_gb`; the old in-use figure fitted two or three)
    "q_block": 32,                   # queries per attention block during prefill
    "cache_q8": True,                # int8 latent cache (a 262k set costs ~0.2 GB/chip instead of 0.3)
    "mhc_kernel": "bf16",            # the mHC sites of decode / verify steps as Pallas kernels (glm53.pallas_mhc: plain
                                     # decode -8 %, DFlash 3 streams -4 % on the v6e); "" = XLA's chain (the numerics
                                     # of the builds before 2026-10-05)
    "bucket_min": 256,               # smallest prefill bucket compiled (short prompts pad to it)
    "min_free_gb": 0.75,             # HBM an admission needs beyond the live buffers: a 262k cache set (0.23 GB/chip), one
                                     # piece's prefill temporaries (~0.22) and the programs of a decode step + a prefill
                                     # piece (~0.25: loaded programs are a cache the runtime evicts when buffers need room)
    "max_queue": 8,                  # waiting requests beyond the streams before a 429
    "max_wait_s": 90.0,              # a request waiting longer than this gets a 503 (clients retry)
    "keepalive_s": 15,               # SSE ping when nothing was streamed for this long (a buffered tool call, a queued
                                     # request): Cloudflare drops a response that is silent for ~100 s
    "think_budget_default": 0,       # thinking tokens before </think> is forced, when the request sets no budget (0 = unlimited)
    "vision": True,                  # load the vision tower (images in both APIs); False saves ~1 min and 0.14 GB/chip
    "vision_max_tokens": 1024,       # 28x28-pixel tokens per image
    "reasoning_effort_default": "low",   # server-side default: low | high (anything else = the template's Max)
    "temperature": 1.0, "top_p": 0.95,   # generation_config defaults
    "max_new_default": 4096,
    "snap_host_gb": 48,              # host RAM for parked contexts (agent sessions that interleave)
    "snap_rows": 1024,               # snapshot row bucket (1024 rows = 8192 tokens sharded)
    "snap_min": 256,                 # contexts shorter than this are re-prefilled instead of parked
    "base_min": 512,                 # a system section at least this long is pinned for new sessions
    "snap_warm_tokens": 32768,       # warm the snapshot programs for contexts up to this many tokens
    "keepalive_min": 480,            # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "idle_min": 15,                  # shut down after this many minutes with no active or queued
                                     # request (0 = serve the full keepalive window regardless)
    "burst_gateway": "",             # optional burst control-plane gateway URL;
    "burst_instance": "",            # when set with a tunnel URL, announce {instance, url,
    "burst_gateway_token": "",       # credential_ref, model} to POST /admin/endpoints so the
    "burst_credential_ref": "session/glm53",  # controller can adopt this session
    "burst_model": "glm5.3-flash",
    "api_key": "",                   # generated if empty
    "ntfy_topic": "",                # optional: publish progress to ntfy.sh/<topic> (launch.py watches it)
    "dflash": "hf:incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a",
                                     # DFlash 2 speculative decoding (one stream alone, and batched for 2-3 streams): the
                                     # drafter as `hf:<repo>[@revision]` (fetched from Hugging Face at start: public, 2.3 GB,
                                     # ~1 min; incoai's drafter is CC BY-NC-ND 4.0, research use only), or its directory /
                                     # attached dataset (config.json + model.safetensors); "" = plain decoding (one stream
                                     # ~79 tok/s instead of ~166 on the v5e-8)
    "dflash_int8": True,             # the drafter's big matrices as int8 (159 instead of 295 MB per chip, same acceptance)
    "dflash_cost": {"4": 16.4},      # ms per DFlash step by block size (Kaggle v5e-8, 2026-10-05: 2.6 tokens per step) = the block sizes compiled:
                                     # {"4": ..., "8": ...} also drafts 8 (2026-10-03: math-like text x1.97 instead of
                                     # x1.64, no gain overall at temperature 1.0; its programs only add warm-up)
    "dflash_prior": {"4": 2.8, "8": 3.3},    # tokens per step a new stream assumes until it has measured its own
    "dflash_rows": {"2": [26.0, 14.9], "3": [35.5, 17.0]},   # DFlash for B streams at once: {"B": [ms of a B-stream
                                     # DFlash step (block = the smallest of dflash_cost), ms of a plain batched step of
                                     # B]} = the B compiled (Kaggle v5e-8, 2026-10-05; served 154 / 168 tok/s aggregate at
                                     # 2 / 3 streams vs plain 114 / 147; ~8 min of cold warm-up each); the scheduler picks
                                     # DFlash or plain per step from these and the streams' tokens per step; {} = plain
    "dflash_draft_temp": 0.7,        # drafts are drawn at this x the request's temperature; the verify accepts them by
                                     # rejection sampling (output distributed exactly as plain sampling)
    "served_model_name": "glm-5.3-flash",
    "tunnel": True,                  # False: no cloudflared (local testing)
    "skip_runtime": False,           # True: no pip installs (local testing)
    "port": 8000,
    "bind": "0.0.0.0",
}
CFG = {**DEFAULTS, **(CFG or {}), **globals().get("CFG_PRESET", {})}
_cfg_file = Path("serve_config.json")            # notebook flow: overrides next to this script
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "glm-" + secrets.token_hex(12)

ENGINE_B64 = ""  # __ENGINE__  (launch.py embeds the glm53 package here; the notebook writes the files instead)

PORT = int(CFG["port"])
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
CACHE_DIR = WORK / "jax_cache"
EXPORT_DIR = WORK / "exported"
SERVE_NAMES = [n for n in [CFG["serve_dataset"], *CFG["serve_fallback"]] if n]
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()
LOG_LINES = []


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    LOG_LINES.append(line)
    print(line, flush=True)


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}", "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # noqa: BLE001
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"   {tag}: rc {r.returncode} in {time.time() - t:.0f}s" + (f" | {(r.stderr or '')[-200:].strip()}" if r.returncode else ""))
    return r.returncode


def mounts_of(names):
    """Dataset names ('owner/slug') -> mounted paths, in the given order (missing ones dropped)."""
    out = []
    for n in names:
        slug = n.split("/")[-1]
        hits = glob.glob(f"/kaggle/input/datasets/{n}") + glob.glob(f"/kaggle/input/{slug}")
        if hits:
            out.append(hits[0])
    return out


def hbm():
    st = jax.devices()[0].memory_stats() or {}
    return st.get("bytes_in_use", 0) / 1e9, (st.get("bytes_limit") or 0) / 1e9


def fail(step, msg):
    """Stop with a plain message (the launcher prints `step` and `tail` of a "failed" phase)."""
    log("   " + msg)
    publish("failed", step=step, tail=msg)
    sys.exit(1)


def sanitize_tpu_env():
    """Kaggle's image derives TPU_WORKER_HOSTNAMES / TPU_WORKER_ADDRS from its cluster metadata, and when that
    lookup fails the variables end up holding the lookup's WARNING text instead of addresses — libtpu then refuses
    to start ("INVALID_ARGUMENT: Error: unexpected worker hostname 'WARNING: could not determine ...'"). A v5e-8
    on Kaggle is one VM holding all 8 chips, so PJRT never needs these variables at all: remove them outright
    rather than guess which values are well-formed."""
    dropped = []
    for name in ("TPU_WORKER_HOSTNAMES", "TPU_WORKER_ADDRS"):
        val = os.environ.pop(name, None)
        if val is not None:
            dropped.append(f"{name}={val.strip()[:70]!r}")
    if dropped:
        log("   removed TPU metadata env vars (a single-VM TPU does not need them, and a broken value stops "
            "libtpu): " + ", ".join(dropped))


def preflight():
    """Look before the slow steps (~20 s): the datasets attached, Internet on (pip, cloudflared) and a real TPU present.
    Kaggle sometimes starts a "TPU" session with no TPU (a CPU-only container, most often on new or not-yet-verified
    accounts): jax then sees one CPU device and the build dies minutes later with a sharding error."""
    sanitize_tpu_env()
    need = list(CFG["expert_datasets"]) + ([] if mounts_of(SERVE_NAMES) else
                                           ([CFG["serve_dataset"]] if CFG["serve_dataset"] and not mounts_of(CFG["base_datasets"]) else list(CFG["base_datasets"])))
    missing = [n for n in need if not mounts_of([n])]
    if missing:
        fail("datasets", f"datasets not attached: {missing}. In the right sidebar, Add Input -> search each name -> attach, then run again.")
    try:
        urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=20).read(1)
    except Exception as e:  # noqa: BLE001
        fail("no-internet", f"no Internet from this session ({str(e)[:120]}): Session options -> Internet ON (a phone-verified "
                            "Kaggle account is needed for that), then run again. The pip packages and the tunnel need it.")
    code = ("import jax\n"
            "try:\n"
            "    d = jax.devices()\n"
            "    print('TPU_CHECK', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
            "except Exception as e:\n"
            "    print('TPU_CHECK 0 none', str(e).replace(chr(10), ' ')[:200])\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        m = re.search(r"TPU_CHECK (\d+) (\S+)(.*)", (r.stdout or "") + (r.stderr or ""))
    except Exception as e:  # noqa: BLE001
        log(f"   (TPU check skipped: {e})"); return
    if m is None:
        log("   (TPU check inconclusive: the image's jax did not answer; continuing)"); return
    n, platform, rest = int(m.group(1)), m.group(2), m.group(3).strip()
    if n == 8 and platform == "tpu":
        log(f"   pre-flight OK: datasets attached, Internet on, 8 TPU chips ({rest})"); return
    if n == 0 and not re.search(r"jellyfish|TPU initialization failed|initialize backend 'tpu'|No TPU|vfio", rest, re.I):
        log(f"   (TPU check inconclusive, continuing: {rest[:160]})"); return
    fail("no-tpu", f"this session has no working TPU: jax sees {n} {platform} device(s) {rest}. Kaggle sometimes starts a TPU "
                   "session without one (most often on new or not-yet-verified accounts); nothing in this notebook can fix "
                   "that. Stop the session and start it again; `import jax; print(jax.device_count())` in a fresh cell must "
                   "print 8 before this script is worth running. Some accounts also need full identity verification (KYC "
                   "via Persona) before Kaggle grants TPU access — phone verification alone is not always enough "
                   "(check kaggle.com/settings).")


# ----------------------------------------------------------------------------- 1. runtime
if not CFG["skip_runtime"]:
    banner(1, "Runtime", "pre-flight, pinned libtpu + pip packages, cloudflared, the engine")
    preflight()
    sh([sys.executable, "-m", "pip", "install", "-q", "safetensors", "huggingface_hub", "transformers>=5.16", "pillow",
        "flatbuffers", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"], "pip packages")
    if CFG["libtpu"]:
        sh([sys.executable, "-m", "pip", "install", "-q", f"libtpu=={CFG['libtpu']}"], f"libtpu=={CFG['libtpu']}")
    if ENGINE_B64 and not ENGINE_B64.startswith("__"):
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ENGINE_B64)), mode="r:gz") as tf:
            tf.extractall(WORK)
        sys.path.insert(0, str(WORK))
        log(f"   engine package extracted to {WORK}/glm53")
    elif not Path("glm53").is_dir():
        sys.exit("no glm53/ package next to this script and nothing embedded — run the notebook's engine cell first")

if CFG["tunnel"] and not CLOUDFLARED.exists():
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
    log("   cloudflared downloaded")

import numpy as np                      # noqa: E402  (after the runtime pins: libtpu must be installed before jax loads)
import jax, jax.numpy as jnp            # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

SERVE_MOUNT = (mounts_of(SERVE_NAMES) or [None])[0]
if SERVE_MOUNT and CFG["serve_dataset"] and not mounts_of([CFG["serve_dataset"]]):
    log(f"   {CFG['serve_dataset']} is not attached: using {SERVE_MOUNT} (its weights fit; its compiled programs are "
        "another build's, so every program compiles)")
BASE_DIR = os.path.join(SERVE_MOUNT, "base") if SERVE_MOUNT and os.path.isdir(os.path.join(SERVE_MOUNT, "base")) else None
if "eng" not in globals():              # (a test harness may pre-set eng/tok/ids/eos/cfg and skip the build)
    cache_src = os.path.join(SERVE_MOUNT, "jax_cache") if SERVE_MOUNT and os.path.isdir(os.path.join(SERVE_MOUNT, "jax_cache")) else None
    if cache_src and not CACHE_DIR.exists():
        t = time.time()
        shutil.copytree(cache_src, CACHE_DIR)
        log(f"   compile cache restored from {cache_src} in {time.time() - t:.0f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
    try:                                 # a process that already initialised the cache elsewhere keeps writing there
        from jax._src import compilation_cache as _cc
        _cc.reset_cache()
    except Exception as e:  # noqa: BLE001
        log(f"   (compile cache reset skipped: {e!r})")
    try:                                 # which cache entries this build loads or writes (the serve dataset's next
        if not hasattr(_cc, "_glm_use"):     # jax_cache/ is exactly those: jax_cache_used.txt in the output)
            _cc._glm_use = {"hit": [], "miss": [], "put": [], "read_s": 0.0}

            def _rec(fn, kind, use):         # (`use` is bound here: the warm-up below reuses the name _cc)
                def wrapped(cache_key, *a, **k):
                    t = time.time()
                    r = fn(cache_key, *a, **k)
                    try:
                        if kind == "put":
                            use["put"].append(cache_key)
                        else:
                            use["hit" if r[0] is not None else "miss"].append(cache_key)
                            use["read_s"] += time.time() - t
                    except Exception:  # noqa: BLE001  (the record must never cost a cache read)
                        pass
                    return r
                return wrapped
            _cc.get_executable_and_time = _rec(_cc.get_executable_and_time, "get", _cc._glm_use)
            _cc.put_executable_and_time = _rec(_cc.put_executable_and_time, "put", _cc._glm_use)
        CACHE_USE = _cc._glm_use
        for _k in ("hit", "miss", "put"):
            CACHE_USE[_k].clear()
        CACHE_USE["read_s"] = 0.0
    except Exception as e:  # noqa: BLE001
        log(f"   (compile cache use is not recorded: {e!r})")
    log(f"   jax {jax.__version__}, {jax.device_count()} devices, compile cache at {CACHE_DIR}")
if CFG["aot"]:
    from glm53 import aot as AOT
    _exp_src = os.path.join(SERVE_MOUNT, "exported") if SERVE_MOUNT and os.path.isdir(os.path.join(SERVE_MOUNT, "exported")) else None
    AOT.configure(EXPORT_DIR, [_exp_src], context=lambda: AOT.engine_context(eng), log=log)
    _n_exp = sum(len(glob.glob(os.path.join(d, "*" + AOT.EXT))) for d in AOT.STORE.read_dirs)
    log(f"   exported programs: {_n_exp} files in {', '.join(AOT.STORE.read_dirs)}"
        + ("" if _n_exp else " (none yet: every program is traced, then exported for the next run)"))

# ----------------------------------------------------------------------------- 2. weights -> the chips
if "eng" not in globals():
    banner(2, "Weights", "3-bit codebook experts + int8 non-expert weights onto the eight chips, ~9 min")
    from glm53 import basestore as BS
    from glm53 import checkpoint as C
    from glm53 import model as M
    from glm53.engine import AXIS
    from glm53 import gguf_reader as G
    from glm53.resident import ResidentFetch, ResidentLayerEngine, pack_layer_from_gguf, hbm_bytes
    from transformers import AutoConfig, AutoTokenizer
    gguf_mounts = mounts_of(CFG["expert_datasets"])
    fp8_mounts = [] if BASE_DIR else mounts_of(CFG["base_datasets"])
    if len(gguf_mounts) < len(CFG["expert_datasets"]) or (not BASE_DIR and len(fp8_mounts) < len(CFG["base_datasets"])):
        fail("datasets", f"datasets missing: found gguf {gguf_mounts}, fp8 {fp8_mounts}, serve {SERVE_MOUNT}; attach "
                         f"{CFG['expert_datasets']} + {CFG['serve_dataset'] or CFG['base_datasets']}")
    hf_dir = BASE_DIR or fp8_mounts[0]
    log(f"   experts from {gguf_mounts}; the rest from {'the base store ' + BASE_DIR if BASE_DIR else 'the FP8 checkpoint'}")
    publish("loading", note="weights")
    hf_cfg = AutoConfig.from_pretrained(hf_dir)
    tc = hf_cfg.text_config
    N_LAYERS = tc.num_hidden_layers
    cfg = M.Cfg.from_hf(tc, dtype=jnp.bfloat16)
    N_DEV = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    tbl_sh = NamedSharding(mesh, P(AXIS))
    gm = G.GGUFModel(G.open_mirror(gguf_mounts))
    t_load = time.time()
    if BASE_DIR:
        store = BS.load(BASE_DIR)
        params = {**store["top"], "layers": []}
        log(f"   base store read in {time.time() - t_load:.0f}s ({store['manifest'].get('n_layers')} layers)")
        r = None
    else:
        r = C.RawShardReader(":".join(fp8_mounts))
        top = C.load_top(r, np.float32)
        params = {"embed": top["embed"].astype(jnp.bfloat16), "norm": top["norm"], "lm_head": top["lm_head"].astype(jnp.bfloat16), "layers": []}
        del top
    qtypes, hbm_total = {}, 0
    for i in range(N_LAYERS):
        ti = time.time()
        if BASE_DIR:
            p = store["layers"][i]
        else:
            p = C.load_layer(r, i, tc.layer_types[i], tc.mlp_layer_types[i], np.float32)
            p = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if a.ndim >= 2 else a, p)
        if tc.mlp_layer_types[i] == "sparse":
            tables, qt = pack_layer_from_gguf(gm, i, N_DEV, threads=16)
            qtypes[i] = qt
            hbm_total += hbm_bytes(tables) * N_DEV
            for k, t in tables.items():                    # each table straight onto the chips: host RAM holds one layer
                p["mlp"][k] = jax.tree.map(lambda a: jax.device_put(a, tbl_sh), t)
            jax.block_until_ready([p["mlp"][k] for k in tables])
            del tables
        params["layers"].append(p)
        if r is not None:
            for f in C.layer_shards(":".join(fp8_mounts), i):
                r.release(f)
        if i % 5 == 0 or i == N_LAYERS - 1:
            log(f"   layer {i + 1}/{N_LAYERS} ({time.time() - ti:.0f}s) | HBM chip0 {hbm()[0]:.2f} GB")
    log(f"   expert tables {hbm_total / 1e9:.1f} GB across {N_DEV} chips; weights read in {(time.time() - t_load) / 60:.1f} min")
    M.MHC_KERNEL = CFG["mhc_kernel"] or None
    fetch = ResidentFetch(qtypes, tc.hidden_size, tc.moe_intermediate_size // N_DEV, out_dtype=jnp.bfloat16, gather_max_rows=64, chunk=8,
                          use_pallas=True, sweep_mode="ragged", tm=32, combine="matmul")
    eng = ResidentLayerEngine(cfg, params, fetch, max_len=CFG["max_len"], layers_per_program=4, layers_per_program_prefill=1,
                              int8_nonexpert=False if BASE_DIR else "all", seq_shard=True, q_block=CFG["q_block"],
                              prefill_piece=CFG["piece"], cache_q8=CFG["cache_q8"])
    del params
    use, lim = hbm()
    log(f"   ENGINE READY in {(time.time() - t_load) / 60:.1f} min: HBM chip0 {use:.2f} / {lim:.2f} GB, context {CFG['max_len']}")
    publish("loaded", hbm_gb=round(use, 2), minutes=round((time.time() - t_load) / 60, 1))
    tok = AutoTokenizer.from_pretrained(hf_dir)
    ids = np.asarray(tok(tok.apply_chat_template([{"role": "user", "content": "Write a haiku about tensor processing units."}],
                                                 tokenize=False, add_generation_prompt=True, reasoning_effort="low"),
                         add_special_tokens=False).input_ids).reshape(1, -1)
    eos = set(json.load(open(os.path.join(hf_dir, "config.json")))["text_config"].get("eos_token_id", [tok.eos_token_id]))

# ----------------------------------------------------------------------------- 3. vision tower
VISION, VISION_FWD = globals().get("VISION"), globals().get("VISION_FWD")
_vp = None
if CFG["vision"] and VISION_FWD is None and "hf_dir" in globals():
    banner(3, "Vision", "the vision tower sharded over the chips")
    from glm53 import vision as VIS
    t = time.time()
    if BASE_DIR and store.get("vision") is not None:
        _vp = store["vision"]
    else:
        _vp = VIS.load_vision(C.RawShardReader(":".join(fp8_mounts)), dtype=np.float32)
        _vp = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if getattr(a, "dtype", None) == np.float32 else a, _vp)
    VISION = VIS.to_sharded(_vp, eng.mesh, dtype=jnp.bfloat16)
    VISION_FWD = VIS.make_forward_sharded(VISION, eng.mesh, dtype=jnp.bfloat16, q_chunk=1024)
    from PIL import Image
    for target in [b for b in (256, 512, 1024, 2048, 4096) if b <= 4 * CFG["vision_max_tokens"]]:
        side = int(target ** 0.5) * 14 - 14
        img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (side, side, 3), dtype=np.uint8))
        patches, grid = VIS.preprocess(img, max_tokens=CFG["vision_max_tokens"])
        n = patches.shape[0]
        bucket = max(256, 1 << (n - 1).bit_length())
        grids = (grid,) if bucket == n else (grid, (1, 2, (bucket - n) // 2))
        if bucket > n:
            patches = np.concatenate([patches, np.zeros((bucket - n, patches.shape[1]), patches.dtype)], 0)
        jax.block_until_ready(VISION_FWD(patches, grids))
    log(f"   vision tower ready in {time.time() - t:.0f}s; HBM chip0 {hbm()[0]:.2f} GB")
if CFG["dump_base"] and not BASE_DIR and "hf_dir" in globals():
    t = time.time()
    BS.dump(str(WORK / "base"), eng.params, vision=_vp, hf_dir=hf_dir, log=log,
            meta={"model": CFG["served_model_name"], "int8_nonexpert": "all", "experts": "resident (not stored)"})
    log(f"   base store dumped in {time.time() - t:.0f}s (this run's output can become the serve dataset)")
del _vp

# ----------------------------------------------------------------------------- 3b. DFlash 2 drafter (optional)
_d = CFG["dflash"]
if _d.startswith("hf:"):                 # fetched from Hugging Face at start (a public repo: no token)
    _repo, _, _rev = _d[3:].partition("@")
    DFLASH_DIR = f"/tmp/models/{_repo.split('/')[-1]}"
    try:
        from huggingface_hub import snapshot_download
        t = time.time()
        snapshot_download(_repo, revision=_rev or None, local_dir=DFLASH_DIR,
                          allow_patterns=["config.json", "model.safetensors"])
        log(f"   DFlash drafter {_repo}: fetched in {time.time() - t:.0f}s")
    except Exception as e:  # noqa: BLE001
        log(f"   DFlash: fetching {_repo} failed ({e!r}) -> plain decoding")
        DFLASH_DIR = None
else:
    DFLASH_DIR = (_d if os.path.isdir(_d) else (mounts_of([_d]) or [None])[0]) if _d else None
    if _d and not DFLASH_DIR:
        log(f"   DFlash: {_d} not found -> plain decoding")
if DFLASH_DIR and getattr(eng, "dflash", None) is None:
    from glm53 import dflash as DF
    t = time.time()
    _dcfg, _dp = DF.load_params(DFLASH_DIR, jnp.bfloat16)
    eng.set_dflash(_dp, _dcfg, int8=CFG["dflash_int8"], temp_scale=CFG["dflash_draft_temp"])
    del _dp
    log(f"   DFlash 2 drafter ({_dcfg.n_layers} layers, block {_dcfg.block}) from {DFLASH_DIR}: "
        f"{eng.dflash.nbytes_per_chip() / 1e6:.0f} MB per chip in {time.time() - t:.0f}s; HBM chip0 {hbm()[0]:.2f} GB")
DFLASH_COST = {int(k): float(v) for k, v in CFG["dflash_cost"].items()}
DFLASH_ROWS = {int(k): tuple(float(x) for x in v) for k, v in CFG["dflash_rows"].items() if 1 < int(k) <= CFG["streams"]}

# ----------------------------------------------------------------------------- 4. the server
from glm53.resident import ResidentLayerEngine as _RLE   # noqa: E402
from glm53.engine import DeviceSampler                    # noqa: E402
from glm53.scheduler import DFlashPolicy, Request, Scheduler, SnapStore   # noqa: E402
from glm53 import vision as VIS                           # noqa: E402

API_KEY = CFG["api_key"]
MAX_NEW_DEFAULT = CFG["max_new_default"]
KEEPALIVE_S = float(CFG["keepalive_s"] or 0)
THINK_BUDGET_DEFAULT = int(CFG["think_budget_default"] or 0)
MAX_STREAMS, MAX_SETS = CFG["streams"], CFG["sets"]
SCHED_PIECE = CFG["sched_piece"]
MIN_FREE_GB, MAX_WAIT_S, MAX_QUEUE = CFG["min_free_gb"], CFG["max_wait_s"], CFG["max_queue"]
VISION_MAX_TOKENS = CFG["vision_max_tokens"]
DEFAULT_EFFORT = CFG["reasoning_effort_default"]
DEFAULT_TEMP, DEFAULT_TOP_P = CFG["temperature"], CFG["top_p"]
MODEL = CFG["served_model_name"]
BUCKET_MIN = CFG["bucket_min"]
if BUCKET_MIN:
    eng.PREFILL_BUCKETS = tuple(b for b in type(eng).PREFILL_BUCKETS if b >= BUCKET_MIN)
STATE = {"url": None, "requests": 0, "tokens": 0, "prefix_hits": 0, "prefix_tokens_reused": 0, "prefill_s": 0.0,
         "snap_hits": 0, "snap_parks": 0, "snap_pins": 0, "snap_entries": 0, "snap_bytes": 0, "snap_s": 0.0,
         "steps": 0, "step_tokens": 0}
T = {k: tok.convert_tokens_to_ids(k) for k in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
                                                "<arg_value>", "</arg_value>", "<|user|>", "<|observation|>", "<|assistant|>",
                                                "<|image|>", "<|begin_of_image|>", "<|end_of_image|>")}
STOP_IDS = set(eos) | {T["<|user|>"], T["<|observation|>"]}

# ---- images
IMG_CACHE = collections.OrderedDict()                  # sha256 -> (n_tokens, embeddings f32 [n, D]); LRU
IMG_CACHE_MAX = 64


def embed_image(img_bytes):
    """Image bytes -> (n_tokens, embeddings [n, D] f32) through the vision tower; cached by content hash. The patch
    count is padded to a power of two with a dummy image segment (its rows are dropped) so few shapes compile."""
    from PIL import Image
    if VISION_FWD is None:
        raise ValueError("this server has no vision tower loaded (config: vision)")
    h = hashlib.sha256(img_bytes).hexdigest()
    if h in IMG_CACHE:
        IMG_CACHE.move_to_end(h)
        return IMG_CACHE[h]
    patches, grid = VIS.preprocess(Image.open(io.BytesIO(img_bytes)), max_tokens=VISION_MAX_TOKENS)
    n = patches.shape[0]
    bucket = max(256, 1 << (n - 1).bit_length())
    grids = (grid,) if bucket == n else (grid, (1, 2, (bucket - n) // 2))
    if bucket > n:
        patches = np.concatenate([patches, np.zeros((bucket - n, patches.shape[1]), patches.dtype)], 0)
    t = time.time()
    out = np.asarray(VISION_FWD(patches, grids), np.float32)[:VIS.n_tokens(grid)]
    STATE["vision_s"] = STATE.get("vision_s", 0.0) + time.time() - t
    STATE["images"] = STATE.get("images", 0) + 1
    IMG_CACHE[h] = (out.shape[0], out)
    while len(IMG_CACHE) > IMG_CACHE_MAX:
        IMG_CACHE.popitem(last=False)
    return IMG_CACHE[h]


def _image_bytes(block):
    """Anthropic image block or OpenAI image_url part -> raw bytes (base64 or data: URL inline; http(s) fetched)."""
    if block.get("type") == "image":
        src = block.get("source", {})
        if src.get("type") == "base64":
            return base64.b64decode(src["data"])
        url = src.get("url", "")
    else:
        u = block.get("image_url", "")
        url = u.get("url", "") if isinstance(u, dict) else u
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith("http://") or url.startswith("https://"):
        host = urllib.parse.urlsplit(url).hostname or ""
        # SSRF guard: the kernel runs on a host with access to internal metadata
        # and cluster services; only fetch public addresses.
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
        if not host or any(ipaddress.ip_address(a).is_private or ipaddress.ip_address(a).is_loopback
                           or ipaddress.ip_address(a).is_link_local or ipaddress.ip_address(a).is_reserved
                           for a in addrs):
            raise ValueError("refusing to fetch image from a non-public address")
        return urllib.request.urlopen(url, timeout=30).read()
    raise ValueError("unsupported image source")


def _img_sig(img_bytes):
    """Negative pseudo token id identifying an image in prompt signatures (all of its tokens carry it)."""
    return -(1 + int(hashlib.sha256(img_bytes).hexdigest()[:8], 16) % (1 << 30))


def _dec(ids):
    return tok.decode([T["<|image|>"] if int(i) < 0 else int(i) for i in ids])


def _feed(sig, imgs):
    """Signature ids (image tokens negative) -> (real ids int32 [n], embeds (idx, vec) or None) for the engine."""
    sig_a, off = sig
    ids = np.where(sig_a < 0, T["<|image|>"], sig_a).astype(np.int32)
    idx = np.nonzero(sig_a < 0)[0]
    if len(idx) == 0:
        return ids, None
    vec = np.concatenate([imgs[int(sig_a[i])][1][off[i]:off[i] + 1] for i in idx], 0)
    return ids, (idx, vec)


def _run_offsets(sig):
    """Per position: index within its run of equal negative ids (0 for text)."""
    sig = np.asarray(sig)
    off = np.zeros(len(sig), np.int64)
    for i in range(1, len(sig)):
        if sig[i] < 0 and sig[i] == sig[i - 1]:
            off[i] = off[i - 1] + 1
    return off


# ---- sampling of the first token (the rest are sampled on the device)
def sample(logits, temperature, top_p, rng, n_cand=2048):
    z = np.asarray(logits, dtype=np.float32).reshape(-1)
    if temperature <= 0:
        return int(z.argmax())
    n_cand = min(n_cand, z.size - 1)
    cand = np.argpartition(-z, n_cand)[:n_cand]
    zc = z[cand] / temperature
    zc = zc - zc.max()
    p = np.exp(zc); p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p); cum = np.cumsum(p[order])
        keep = order[: max(1, int((cum <= top_p).sum()) + 1)]
        q = np.zeros_like(p); q[keep] = p[keep]; p = q / q.sum()
    return int(cand[rng.choice(n_cand, p=p)])


BASE_MIN, SNAP_HOST_GB = CFG["base_min"], CFG["snap_host_gb"]
SNAP_ROWS, SNAP_MIN, SNAP_WARM = CFG["snap_rows"], CFG["snap_min"], CFG["snap_warm_tokens"]


def system_end(prompt):
    """Index of the first <|user|> token = end of the rendered system section (system prompt + tools)."""
    try:
        return prompt.index(T["<|user|>"])
    except ValueError:
        return 0


def _match_len(live_ids, pos, prompt, quiet=False):
    """Longest reuse of the live context (ids[:pos]) for `prompt`: (k, fed) where prompt[k:] must still be prefilled and
    `fed` = the ids the engine will have seen after that, or (0, None). Handles a dropped thinking block (the template
    renders `<think></think>` where the live context holds the reasoning) and re-tokenisation drift near the boundary."""
    THINK, END = T["<|assistant|>"], T["</think>"]
    i = j = 0
    while True:
        n = min(pos - i, len(prompt) - j)
        if n > 0:
            neq = np.asarray(live_ids[i:i + n]) != np.asarray(prompt[j:j + n])
            d = int(np.argmax(neq)) if neq.any() else n
            i += d; j += d
        if i >= pos:
            return (j, list(live_ids[:pos]) + list(prompt[j:])) if j <= len(prompt) else (0, None)
        if j >= len(prompt):
            return 0, None
        if i > 0 and live_ids[i - 1] == T["<think>"] and prompt[j] == END and END in live_ids[i:pos]:
            i = live_ids.index(END, i) + 1; j += 1
            continue
        if pos - i > 256:
            return 0, None
        a = max(0, i - 32); ja = j - (i - a)
        t_live = _dec(live_ids[a:pos])
        for k in range(max(ja + 1, j - 24), min(len(prompt), j + (pos - i) + 24) + 1):
            if _dec(prompt[ja:k]) == t_live:
                return k, list(live_ids[:pos]) + list(prompt[k:])
        return 0, None


RNG = np.random.default_rng()


def sched_feed(req, a, b):
    return _feed((req.sig[a:b], req.off[a:b]), req.imgs or {})


_drop = [k for k in eng._progs if isinstance(k, tuple) and k[0] == "group" and k[3] == 1 and k[5]]   # batch-1 loop programs: unused here
for k in _drop:
    del eng._progs[k]
SNAPS = SnapStore(eng, int(SNAP_HOST_GB * 1e9), SNAP_ROWS, match_len=_match_len, log=log, state=STATE)
SCHED = Scheduler(eng, STOP_IDS, MAX_STREAMS, MAX_SETS, feed=sched_feed, match_len=_match_len, system_end=system_end,
                  first_sample=lambda z, t, p: sample(z, t, p, RNG), snaps=SNAPS, log=log, state=STATE,
                  base_min=BASE_MIN, snap_min=SNAP_MIN, piece=SCHED_PIECE, min_free_gb=MIN_FREE_GB, max_wait_s=MAX_WAIT_S,
                  spec=DFlashPolicy(DFLASH_COST, {int(k): float(v) for k, v in CFG["dflash_prior"].items()
                                                  if int(k) in DFLASH_COST}, rows=DFLASH_ROWS)
                  if getattr(eng, "dflash", None) is not None else None)


class QueueFull(Exception):
    """Too many requests in flight (-> 429)."""


class ClientGone(Exception):
    """The client closed the connection mid-response."""


_IDLE = object()


def generate(prompt, max_new, temperature, top_p, on_token=None, imgs=None, rid=None, on_idle=None, budget=None):
    """One request through the scheduler -> (out ids, prefill_s, decode_s, reused, reason) with reason "stop" | "length"
    | "stop_sequence" (`on_token` returned True: cancelled there) | "cancelled". Tokens reach `on_token` on THIS thread;
    `on_idle` is called when no token arrived for KEEPALIVE_S (a queued request); `budget` = Request.budget."""
    prompt = [int(t) for t in prompt]
    if MAX_QUEUE and len(SCHED.pending) + len(SCHED.active) >= MAX_STREAMS + MAX_QUEUE:
        raise QueueFull(f"{len(SCHED.active)} requests running and {len(SCHED.pending)} waiting; retry later")
    q = queue.Queue()
    req = Request(prompt, max_new, temperature, top_p, imgs=imgs, rid=rid, budget=budget,
                  on_token=q.put if on_token else None, on_done=(lambda: q.put(None)) if on_token else None)
    req.sig = np.asarray(prompt, np.int64)
    req.off = _run_offsets(req.sig) if imgs else np.zeros(len(prompt), np.int64)
    SCHED.submit(req)
    stopped = False
    if on_token:
        while True:
            try:
                t = q.get(timeout=KEEPALIVE_S or None)
            except queue.Empty:
                t = _IDLE
            if t is None:
                break
            if stopped:
                continue
            try:
                if t is _IDLE:
                    if on_idle is not None:
                        on_idle()
                elif on_token(t):
                    stopped = True
                    req.cancel()
            except Exception:
                req.cancel()
                raise
    req.done.wait()
    if req.error is not None:
        raise req.error
    return req.out, req.prefill_s, req.decode_s, req.reused, ("stop_sequence" if stopped else req.stop_reason)


# ---- output parsing (token level)
class StopFilter:
    """`stop_sequences` on the TEXT events: `feed(text) -> (emit, hit)` holds back a tail that could begin a stop
    sequence; at a match returns the text before it and the sequence; `flush()` releases the held tail."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.buf, self.hit = "", None

    def feed(self, text):
        if not self.stops:
            return text, None
        self.buf += text
        best = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, s)
        if best is not None:
            out, self.buf, self.hit = self.buf[:best[0]], "", best[1]
            return out, best[1]
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), keep, -1):
                if self.buf.endswith(s[:k]):
                    keep = k; break
        cut = len(self.buf) - keep
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, None

    def flush(self):
        out, self.buf = self.buf, ""
        return out


def parse_tool_call(body, tools):
    """body = tokens between <tool_call> and </tool_call>."""
    name_end = body.index(T["<arg_key>"]) if T["<arg_key>"] in body else len(body)
    name = tok.decode(body[:name_end]).strip()
    schema = {}
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            schema = (f.get("parameters") or f.get("input_schema") or {}).get("properties", {}) or {}
    args, i = {}, name_end
    while i < len(body) and T["<arg_key>"] in body[i:]:
        a = body.index(T["<arg_key>"], i) + 1
        b = body.index(T["</arg_key>"], a) if T["</arg_key>"] in body[a:] else len(body)
        key = tok.decode(body[a:b]).strip()
        c = body.index(T["<arg_value>"], b) + 1 if T["<arg_value>"] in body[b:] else len(body)
        d = body.index(T["</arg_value>"], c) if T["</arg_value>"] in body[c:] else len(body)
        raw = tok.decode(body[c:d])
        typ = schema.get(key, {}).get("type")
        if typ == "string":
            val = raw
        else:
            try:
                val = json.loads(raw)
            except Exception:  # noqa: BLE001
                val = raw
        args[key] = val
        i = d + 1
    return {"id": "call_" + uuid.uuid4().hex[:12], "name": name, "input": args}


class TokenStream:
    """Incremental token -> (kind, text/tool) events: kind in {"thinking", "text", "tool"}; partial multibyte text
    is held back until it decodes cleanly; tool-call tokens are buffered until </tool_call>."""

    def __init__(self, tools, mode="thinking"):
        self.tools, self.mode, self.buf, self.tool_buf = tools, mode, [], []

    def feed(self, t):
        if t in STOP_IDS:
            return self.flush()
        if self.mode == "tool":
            if t == T["</tool_call>"]:
                self.mode = "text"; call = parse_tool_call(self.tool_buf, self.tools); self.tool_buf = []
                return [("tool", call)]
            self.tool_buf.append(t); return []
        if t == T["</think>"] and self.mode == "thinking":
            ev = self.flush(); self.mode = "text"; return ev
        if t == T["<tool_call>"] and self.mode == "text":
            ev = self.flush(); self.mode = "tool"; return ev
        self.buf.append(t)
        piece = tok.decode(self.buf)
        if piece.endswith("�"):
            return []
        self.buf = []
        return [(self.mode, piece)] if piece else []

    def flush(self):
        if not self.buf:
            return []
        piece = tok.decode(self.buf); self.buf = []
        return [(self.mode, piece)] if piece else []


THINK_WRAP = tok("\n\nMy thinking budget is used up, so I will give the answer directly now.\n", add_special_tokens=False).input_ids + [T["</think>"]]


def run_request(ids, max_new, temperature, top_p, imgs, rid, tools, stops=None, prefix=(), on_event=None, raw=False,
                think_budget=None):
    """One request end to end: `prefix` ids (a forced tool call) are fed to the parser first, then the generated tokens;
    events (kind, value) go to `on_event` as they are parsed (stop sequences applied to the text), plus ("ping", None)
    when nothing was emitted for KEEPALIVE_S (a tool call is buffered until it closes; a queued request waits): the
    tunnel drops a response that stays silent for ~100 s. `think_budget`: thinking tokens after which `</think>` is
    forced (a short wrap-up line first) so the answer always comes. Returns
    {reasoning, text, calls, out_len, stop, stop_sequence, tp, tg, reused}."""
    ts, sf = TokenStream(tools, "text" if raw else "thinking"), StopFilter(stops)
    res = {"reasoning": "", "text": "", "calls": [], "n": 0}
    last = [time.time()]

    def tick():
        if on_event is not None and KEEPALIVE_S and time.time() - last[0] >= KEEPALIVE_S:
            last[0] = time.time()
            on_event("ping", None)

    def emit(kind, val):
        last[0] = time.time()
        if kind == "thinking":
            res["reasoning"] += val
        elif kind == "text":
            res["text"] += val
        else:
            res["calls"].append(val)
        if on_event is not None:
            on_event(kind, val)

    def handle(t):
        for kind, val in ts.feed(t):
            if kind == "text":
                out, hit = sf.feed(val)
                if out:
                    emit("text", out)
                if hit is not None:
                    return True
            else:
                held = sf.flush()
                if held:
                    emit("text", held)
                emit(kind, val)
        return False

    for t in prefix:
        handle(t)

    def on_token(t):
        res["n"] += 1
        if handle(t):
            return True
        tick()
        return False
    full = list(ids) + list(prefix)
    budget = (int(think_budget), T["</think>"], THINK_WRAP) if think_budget and not raw and full and full[-1] == T["<think>"] else None
    try:
        out, tp, tg, reused, reason = generate(full, max_new, temperature, top_p, on_token, imgs, rid, on_idle=tick, budget=budget)
    except ClientGone as e:
        e.n = res["n"]
        raise
    if reason != "stop_sequence":
        held = sf.flush()
        if held:
            emit("text", held)
    return {"reasoning": res["reasoning"], "text": res["text"], "calls": res["calls"], "out_len": res["n"],
            "stop": reason, "stop_sequence": sf.hit, "tp": tp, "tg": tg, "reused": reused}


def forced_prefix(choice):
    """tool_choice -> generation prefix ids: "any" forces a tool call, a name forces that tool (the template's own
    rendering of a tool turn: `<think></think><tool_call>name...`, so the next turn's prompt matches the live context)."""
    if choice in (None, "none"):
        return []
    ids = [T["</think>"], T["<tool_call>"]]
    if choice != "any":
        ids += tok(choice, add_special_tokens=False).input_ids
    return ids


def tool_choice_of(req, api):
    """-> None (auto) | "none" | "any" | tool name, from an Anthropic or OpenAI request."""
    tc = req.get("tool_choice")
    if tc is None or not req.get("tools"):
        return None
    if api == "anthropic":
        kind = tc.get("type", "auto") if isinstance(tc, dict) else str(tc)
        return {"auto": None, "none": "none", "any": "any"}.get(kind, tc.get("name") if isinstance(tc, dict) else None)
    if isinstance(tc, str):
        return {"none": "none", "required": "any"}.get(tc)
    return (tc.get("function") or {}).get("name") or None


def stops_of(req):
    s = req.get("stop_sequences", req.get("stop"))
    if not s:
        return []
    return [s] if isinstance(s, str) else [str(x) for x in s if x]


# ---- request conversion
STRIP_PATTERNS = [re.compile(p) for p in [
    r"\s*<total_tokens>[^<]*</total_tokens>\s*",            # Claude Code's per-request token budget reminder
    r"x-anthropic-billing-header:[^\n]*\n?",                # Claude Code's per-session billing metadata (system role)
]]


def clean(text):
    for pat in STRIP_PATTERNS:
        text = pat.sub("", text)
    return text


def anthropic_to_messages(req):
    """Anthropic Messages request -> (chat-template messages, tools in OpenAI function form)."""
    msgs = []
    sysm = req.get("system")
    if sysm:
        text = sysm if isinstance(sysm, str) else "\n".join(b.get("text", "") for b in sysm if b.get("type") == "text")
        msgs.append({"role": "system", "content": clean(text)})
    for m in req.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            c = clean(c)
            if c.strip() or m["role"] != "system":
                msgs.append({"role": m["role"], "content": c})
            continue
        if m["role"] == "system":
            text = clean("\n".join(b.get("text", "") for b in c if b.get("type") == "text"))
            if text.strip():
                msgs.append({"role": "system", "content": text})
            continue
        if m["role"] == "user":
            items, results = [], []
            for b in c:
                if b.get("type") == "text":
                    if clean(b["text"]).strip():
                        items.append({"type": "text", "text": clean(b["text"])})
                elif b.get("type") == "image":
                    items.append({"type": "image", "_img": _image_bytes(b)})
                elif b.get("type") == "tool_result":
                    rc = b.get("content", "")
                    if isinstance(rc, list):
                        parts = [{"type": "text", "text": x.get("text", "")} if x.get("type") == "text"
                                 else {"type": "image", "_img": _image_bytes(x)} for x in rc if x.get("type") in ("text", "image")]
                        rc = parts if any(p["type"] == "image" for p in parts) else "\n".join(p["text"] for p in parts)
                    results.append({"role": "tool", "content": rc if isinstance(rc, list) else str(rc),
                                    "tool_call_id": b.get("tool_use_id", "")})
            msgs.extend(results)
            if items:
                msgs.append({"role": "user", "content": _content(items)})
        else:
            texts, thinks, calls = [], [], []
            for b in c:
                if b.get("type") == "text":
                    texts.append(b["text"])
                elif b.get("type") == "thinking":
                    thinks.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b["name"], "arguments": b.get("input", {})}})
            am = {"role": "assistant", "content": "\n".join(texts)}
            if thinks:
                am["reasoning_content"] = "\n".join(thinks)
            if calls:
                am["tool_calls"] = calls
            msgs.append(am)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema", {"type": "object", "properties": {}})}}
             for t in req.get("tools", []) if t.get("name")]
    return msgs, tools


def openai_messages(req):
    msgs = []
    for m in req["messages"]:
        m = dict(m)
        if isinstance(m.get("content"), list):
            items = [{"type": "text", "text": p.get("text", "")} if p.get("type") == "text"
                     else {"type": "image", "_img": _image_bytes(p)} for p in m["content"] if p.get("type") in ("text", "image_url")]
            m["content"] = _content(items)
        if m.get("content") is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except Exception:  # noqa: BLE001
                    fn["arguments"] = {"_raw": fn["arguments"]}
        msgs.append(m)
    return msgs, req.get("tools") or None


def _content(items):
    if any(it["type"] == "image" for it in items):
        return items
    return "\n".join(it["text"] for it in items)


def _images_of(msgs):
    out = []
    for m in msgs:
        if isinstance(m.get("content"), list):
            out += [it["_img"] for it in m["content"] if it.get("type") == "image"]
    return out


def render(msgs, tools, effort):
    """-> (signature ids, imgs): token ids with every <|image|> expanded to the image's token count, those tokens
    replaced by the image's negative signature id; imgs maps signature -> (n_tokens, embeddings)."""
    text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    ids = tok(text, add_special_tokens=False).input_ids
    images = _images_of(msgs)
    if not images:
        return ids, {}
    imgs, out, n_img = {}, [], 0
    for t in ids:
        if t != T["<|image|>"]:
            out.append(t); continue
        if n_img >= len(images):
            raise ValueError("more <|image|> tokens than images")
        b = images[n_img]; n_img += 1
        v = _img_sig(b)
        imgs[v] = embed_image(b)
        out += [v] * imgs[v][0]
    if n_img != len(images):
        raise ValueError("fewer <|image|> tokens than images")
    return out, imgs


def think_budget_of(req):
    """Thinking tokens before `</think>` is forced: Anthropic `thinking.budget_tokens`, OpenAI-style `thinking_budget`
    or `reasoning.max_tokens`, else the server default; None = unlimited."""
    th = req.get("thinking") or {}
    b = th.get("budget_tokens") if th.get("type") == "enabled" else None
    if b is None:
        b = req.get("thinking_budget") or (req.get("reasoning") or {}).get("max_tokens")
    b = int(b or THINK_BUDGET_DEFAULT or 0)
    return b if b > 0 else None


def effort_of(req, default=DEFAULT_EFFORT):
    e = (req.get("chat_template_kwargs") or {}).get("reasoning_effort") or req.get("reasoning_effort")
    th = req.get("thinking") or {}
    if not e and th.get("type") == "enabled":
        e = "high" if int(th.get("budget_tokens", 0) or 0) >= 8192 else "low"
    if not e and th.get("type") == "adaptive":
        e = default
    if not e and th.get("type") == "disabled":
        e = "low"
    return e or default


# ---- HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # one request per connection: SSE responses carry no framing for keep-alive

    def log_message(self, *a):
        pass

    def _json(self, code, obj, headers=()):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self._raw(body)

    def _raw(self, b):
        try:
            self.wfile.write(b if isinstance(b, bytes) else b.encode()); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise ClientGone() from None

    def _auth(self):
        if not API_KEY:
            return True
        h = self.headers.get("Authorization", ""); k = self.headers.get("x-api-key", "")
        expected = API_KEY.encode()
        return hmac.compare_digest(h.encode(), f"Bearer {API_KEY}".encode()) or hmac.compare_digest(k.encode(), expected)

    def _sse(self, obj, event=None):
        self._raw((f"event: {event}\n" if event else "") + f"data: {json.dumps(obj)}\n\n")

    def _stream_error(self, code, kind, message):
        """Finish an already-started SSE response without writing a second HTTP status line."""
        protocol = getattr(self, "_stream_format", None)
        if protocol == "anthropic":
            error_type = "rate_limit_error" if code == 429 else "overloaded_error" if code == 503 else kind
            self._sse({"type": "error", "error": {"type": error_type, "message": message}}, "error")
            self._stream_format = None
            return
        if protocol == "openai":
            self._sse({"error": {"message": message, "type": kind, "code": kind}})
            self._raw(b"data: [DONE]\n\n")
            self._stream_format = None
            return
        self._json(code, {"error": {"type": kind, "message": message}})

    def do_GET(self):
        if not self._auth():
            return self._json(401, {"error": {"type": "authentication_error", "message": "bad api key"}})
        if self.path.startswith("/health"):
            return self._json(200, {"status": "ok", "model": MODEL, "layers": cfg.n_layers, "max_len": eng.max_len, "max_streams": MAX_STREAMS,
                                    "max_sets": MAX_SETS, "max_queue": MAX_QUEUE, "piece": SCHED.piece, "buckets": list(eng.PREFILL_BUCKETS),
                                    "active": len(SCHED.active), "pending": len(SCHED.pending),
                                    "live_contexts": len(SCHED.live), "hbm_free_gb": SCHED.free_gb(),
                                    "hbm_free_in_use_gb": SCHED.in_use_free_gb(), **STATE})
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            return self._json(401, {"error": {"type": "authentication_error", "message": "bad api key"}})
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": str(e)})
        try:
            if self.path.startswith("/v1/messages/count_tokens"):
                msgs, tools = anthropic_to_messages(req)
                return self._json(200, {"input_tokens": len(render(msgs, tools, effort_of(req))[0])})
            if self.path.startswith("/v1/messages"):
                return self.anthropic(req)
            if self.path.startswith("/v1/chat/completions"):
                return self.openai_chat(req)
            if self.path.startswith("/v1/completions"):
                return self.openai_text(req)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, ClientGone):
                return log(f"client disconnected: {getattr(self, 'rid', '?')} after {getattr(e, 'n', '?')} tokens (stream cancelled)")
            if isinstance(e, QueueFull):
                STATE["rejected"] = STATE.get("rejected", 0) + 1
                if getattr(self, "_stream_format", None):
                    return self._stream_error(429, "rate_limit_error", str(e))
                return self._json(429, {"error": {"type": "rate_limit_error", "message": str(e)}}, [("Retry-After", "5")])
            import traceback; log("request failed:", traceback.format_exc()[-1500:])
            code, kind = (503, "overloaded_error") if isinstance(e, TimeoutError) else \
                         (400, "invalid_request_error") if isinstance(e, ValueError) else (500, "api_error")
            try:
                if getattr(self, "_stream_format", None):
                    return self._stream_error(code, kind, str(e))
                return self._json(code, {"error": {"type": kind, "message": str(e)}})
            except Exception:  # noqa: BLE001
                return
        self._json(404, {"error": "not found"})

    # ---- OpenAI
    def openai_chat(self, req):
        msgs, tools = openai_messages(req)
        choice = tool_choice_of(req, "openai")
        ids, imgs = render(msgs, None if choice == "none" else tools, effort_of(req))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or MAX_NEW_DEFAULT)
        temperature = float(req.get("temperature", DEFAULT_TEMP)); top_p = float(req.get("top_p", DEFAULT_TOP_P))
        rid = self.rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        STATE["requests"] += 1
        kw = dict(tools=tools, stops=stops_of(req), prefix=forced_prefix(choice), think_budget=think_budget_of(req))
        finish = lambda r: "tool_calls" if r["calls"] else ("length" if r["stop"] == "length" else "stop")
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            self._stream_format = "openai"
            nt = [0]

            def chunk(delta, finish=None):
                self._sse({"id": rid, "object": "chat.completion.chunk", "model": MODEL,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

            def on_event(kind, val):
                if kind == "ping":
                    self._raw(": keep-alive\n\n")               # an SSE comment: OpenAI clients skip it
                elif kind == "thinking":
                    chunk({"reasoning_content": val})
                elif kind == "text":
                    chunk({"content": val})
                else:
                    chunk({"tool_calls": [{"index": nt[0], "id": val["id"], "type": "function",
                                           "function": {"name": val["name"], "arguments": json.dumps(val["input"])}}]})
                    nt[0] += 1
            r = run_request(ids, max_new, temperature, top_p, imgs, rid, on_event=on_event, **kw)
            chunk({}, finish(r))
            self._raw(b"data: [DONE]\n\n")
            self._stream_format = None
        else:
            r = run_request(ids, max_new, temperature, top_p, imgs, rid, **kw)
            msg = {"role": "assistant", "content": r["text"]}
            if r["reasoning"]:
                msg["reasoning_content"] = r["reasoning"]
            if r["calls"]:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["input"])}} for c in r["calls"]]
            self._json(200, {"id": rid, "object": "chat.completion", "model": MODEL,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish(r)}],
                             "usage": {"prompt_tokens": len(ids), "completion_tokens": r["out_len"],
                                       "prefill_s": round(r["tp"], 2), "decode_tok_s": round(r["out_len"] / max(r["tg"], 1e-6), 2)}})
        self._done(rid, ids, r)

    def openai_text(self, req):
        ids = tok(req.get("prompt", ""), add_special_tokens=False).input_ids
        max_new = int(req.get("max_tokens") or MAX_NEW_DEFAULT)
        temperature = float(req.get("temperature", DEFAULT_TEMP)); top_p = float(req.get("top_p", DEFAULT_TOP_P))
        rid = self.rid = "cmpl-" + uuid.uuid4().hex[:12]
        STATE["requests"] += 1
        r = run_request(ids, max_new, temperature, top_p, None, rid, None, stops=stops_of(req), raw=True)
        self._json(200, {"id": rid, "object": "text_completion", "model": MODEL,
                         "choices": [{"index": 0, "text": r["text"], "finish_reason": "length" if r["stop"] == "length" else "stop"}],
                         "usage": {"prompt_tokens": len(ids), "completion_tokens": r["out_len"]}})
        self._done(rid, ids, r)

    # ---- Anthropic
    def anthropic(self, req):
        msgs, tools = anthropic_to_messages(req)
        choice = tool_choice_of(req, "anthropic")
        ids, imgs = render(msgs, None if choice == "none" else tools, effort_of(req))
        max_new = int(req.get("max_tokens") or MAX_NEW_DEFAULT)
        temperature = float(req.get("temperature", DEFAULT_TEMP)); top_p = float(req.get("top_p", DEFAULT_TOP_P))
        rid = self.rid = "msg_" + uuid.uuid4().hex[:16]
        STATE["requests"] += 1
        kw = dict(tools=tools, stops=stops_of(req), prefix=forced_prefix(choice), think_budget=think_budget_of(req))
        stop_of = lambda r: "tool_use" if r["calls"] else {"length": "max_tokens", "stop_sequence": "stop_sequence"}.get(r["stop"], "end_turn")
        if req.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            self._stream_format = "anthropic"
            self._sse({"type": "message_start", "message": {"id": rid, "type": "message", "role": "assistant", "model": MODEL,
                                                             "content": [], "stop_reason": None, "stop_sequence": None,
                                                             "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
            st = {"idx": -1, "open": None}

            def open_block(kind):
                st["idx"] += 1; st["open"] = kind
                blk = {"type": "thinking", "thinking": ""} if kind == "thinking" else {"type": "text", "text": ""}
                self._sse({"type": "content_block_start", "index": st["idx"], "content_block": blk}, "content_block_start")

            def close_block():
                if st["open"] is not None:
                    self._sse({"type": "content_block_stop", "index": st["idx"]}, "content_block_stop"); st["open"] = None

            def on_event(kind, val):
                if kind == "ping":
                    self._sse({"type": "ping"}, "ping")            # the Anthropic API's own keep-alive event
                elif kind in ("thinking", "text"):
                    if st["open"] != kind:
                        close_block(); open_block(kind)
                    d = {"type": "thinking_delta", "thinking": val} if kind == "thinking" else {"type": "text_delta", "text": val}
                    self._sse({"type": "content_block_delta", "index": st["idx"], "delta": d}, "content_block_delta")
                else:
                    close_block(); st["idx"] += 1
                    self._sse({"type": "content_block_start", "index": st["idx"],
                               "content_block": {"type": "tool_use", "id": val["id"], "name": val["name"], "input": {}}}, "content_block_start")
                    self._sse({"type": "content_block_delta", "index": st["idx"],
                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(val["input"])}}, "content_block_delta")
                    self._sse({"type": "content_block_stop", "index": st["idx"]}, "content_block_stop")
            r = run_request(ids, max_new, temperature, top_p, imgs, rid, on_event=on_event, **kw)
            close_block()
            self._sse({"type": "message_delta", "delta": {"stop_reason": stop_of(r), "stop_sequence": r["stop_sequence"]},
                       "usage": {"output_tokens": r["out_len"]}}, "message_delta")
            self._sse({"type": "message_stop"}, "message_stop")
            self._stream_format = None
        else:
            r = run_request(ids, max_new, temperature, top_p, imgs, rid, **kw)
            content = []
            if r["reasoning"]:
                content.append({"type": "thinking", "thinking": r["reasoning"], "signature": ""})
            if r["text"].strip() or not r["calls"]:
                content.append({"type": "text", "text": r["text"]})
            content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]} for c in r["calls"]]
            self._json(200, {"id": rid, "type": "message", "role": "assistant", "model": MODEL, "content": content,
                             "stop_reason": stop_of(r), "stop_sequence": r["stop_sequence"],
                             "usage": {"input_tokens": len(ids), "output_tokens": r["out_len"]}})
        self._done(rid, ids, r)

    def _done(self, rid, ids, r):
        log(f"req {rid}: {len(ids)} prompt tok ({r['reused']} reused), {r['out_len']} new, prefill {r['tp']:.2f}s, "
            f"{r['out_len'] / max(r['tg'], 1e-6):.1f} tok/s, {r['stop']}")


# ----------------------------------------------------------------------------- 5. warm-up, serve, tunnel
banner(4, "Warm-up", "prefill buckets, batched decode programs, snapshot programs")
t_warm = time.time()
_u = [hbm()[0]]


def grew():
    """HBM the last warm-up stage kept (its programs: compiled code is loaded onto the chips at a program's first run
    and stays for the session), in MB."""
    u = hbm()[0]
    d, _u[0] = u - _u[0], u
    return f"+{d * 1e3:.0f} MB"


for b in [b for b in eng.PREFILL_BUCKETS if b <= eng.prefill_piece]:
    t = time.time()
    seq = np.concatenate([ids] * (b // ids.shape[1] + 1), 1)[:, :b]
    lg, cc, _ = eng.prefill(seq); jax.block_until_ready(lg); del cc
    log(f"   prefill bucket {b}: {time.time() - t:.0f}s, {grew()}")
    publish("compiling", what=f"prefill {b}", secs=int(time.time() - t))
for B in range(1, MAX_STREAMS + 1):
    t = time.time()
    _sets = [eng.alloc_caches(1) for _ in range(B)]
    _samp = DeviceSampler([1.0] * B, [0.95] * B, seed=0)
    _ids, _lg, _sets, _pos = eng.decode_rows(np.full((B,), 0, np.int32), _sets, [0] * B, _samp)
    jax.block_until_ready(_ids); del _sets, _lg, _ids, _pos
    log(f"   batched decode B={B}: {time.time() - t:.0f}s, {grew()}")
    publish("compiling", what=f"decode B={B}", secs=int(time.time() - t))
if getattr(eng, "dflash", None) is not None:          # draft (every block-size pair), verify + capture, head, settle
    for _T in sorted(DFLASH_COST):
        t = time.time()
        _lg, _cc, _p = eng.prefill(ids)
        _st = eng.dflash_start(np.array([0]), _cc, _p, T=_T)
        for _nt in sorted(DFLASH_COST):
            _e, _st = eng.dflash_step(_st, DeviceSampler(1.0, 0.95, seed=0), tuple(sorted(STOP_IDS)), next_T=_nt)
        _st = eng.dflash_settle(_st)                  # the stream's end: the verify run ahead undone (its program)
        del _lg, _cc, _st
        eng.dflash_end()                              # (the last step's arrays, kept for a deferred release)
        log(f"   DFlash block {_T}: {time.time() - t:.0f}s, {grew()}")
        publish("compiling", what=f"dflash T={_T}", secs=int(time.time() - t))
    for B in sorted(DFLASH_ROWS):                     # B streams at once: batched verify, draft, head, rollback
        t = time.time()
        _sets, _tok, _pp = [], [], []
        for _ in range(B):
            _lg, _cc, _p = eng.prefill(ids); _sets.append(_cc); _tok.append(0); _pp.append(_p)
        _st = eng.dflash_start_rows(_tok, _sets, _pp, T=min(DFLASH_COST),
                                    sampler=DeviceSampler([1.0] * B, [0.95] * B, seed=0))
        _sm = DeviceSampler([1.0] * B, [0.95] * B, seed=0)
        for _ in range(2):
            _e, _st = eng.dflash_step_rows(_st, _sm, tuple(sorted(STOP_IDS)))
        _st = eng.dflash_settle_rows(_st)             # (the state's end: every row's undo)
        del _lg, _sets, _st
        eng.dflash_end()
        log(f"   DFlash {B} streams: {time.time() - t:.0f}s, {grew()}")
        publish("compiling", what=f"dflash B={B}", secs=int(time.time() - t))
n_shard = max(eng.lcfg.seq_shard, 1)
for n_tok in range(SNAP_ROWS * n_shard, SNAP_WARM + 1, SNAP_ROWS * n_shard):
    cc = eng.alloc_caches(1)
    hs = eng.snapshot_to_host(eng.snapshot_prefix(cc, n_tok, rows_bucket=SNAP_ROWS)); del cc
    cc = eng.restore_prefix(eng.snapshot_from_host(hs)); jax.block_until_ready(cc[0]["state"] if "state" in cc[0] else cc[0]["c"]); del cc, hs
log(f"   snapshot programs: {grew()}")
use, lim = hbm()
log(f"   warm-up done in {(time.time() - t_warm) / 60:.1f} min; HBM chip0 {use:.2f} / {lim:.2f} GB")
if CACHE_DIR.exists():
    n_files = sum(1 for _ in CACHE_DIR.iterdir()); size = sum(f.stat().st_size for f in CACHE_DIR.iterdir()) / 1e9
    log(f"   compile cache: {n_files} entries, {size:.2f} GB in {CACHE_DIR}")
if globals().get("CACHE_USE") is not None:
    _used = sorted(set(CACHE_USE["hit"]) | set(CACHE_USE["put"]))
    (WORK / "jax_cache_used.txt").write_text("\n".join(_used) + "\n")
    log(f"   compile cache use: {len(set(CACHE_USE['hit']))} programs loaded from it ({CACHE_USE['read_s']:.0f}s), "
        f"{len(set(CACHE_USE['put']))} compiled and stored, {len(set(CACHE_USE['miss']) - set(CACHE_USE['put']))} compiled only "
        f"(small); the {len(_used)} entries of this build are listed in {WORK / 'jax_cache_used.txt'}")
if CFG["aot"]:
    _fs = list(EXPORT_DIR.glob("*" + AOT.EXT))
    log(f"   {AOT.summary()}; {len(_fs)} files, {sum(f.stat().st_size for f in _fs) / 1e9:.2f} GB in {EXPORT_DIR}")
publish("warmed", minutes=round((time.time() - t_warm) / 60, 1), hbm_gb=round(use, 2))

SCHED.start()
srv = ThreadingHTTPServer((CFG["bind"], PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
log(f"   HTTP server on :{PORT}")

banner(5, "Tunnel", "a public cloudflared URL")
url = None
TUN = [None]


def start_tunnel(attempts=3, wait_s=90):
    """A cloudflared quick tunnel -> its public URL, or None. Registration sometimes fails or hangs (seen in our runs):
    an attempt that prints no URL within `wait_s` is killed and retried."""
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for i in range(attempts):
        tun = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        q = queue.Queue()
        threading.Thread(target=lambda: [q.put(l) for l in iter(tun.stdout.readline, "")], daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                line = q.get(timeout=5)
            except queue.Empty:
                if tun.poll() is not None:
                    break
                continue
            m = pat.search(line)
            if m:
                TUN[0] = tun
                return m.group(0)
        tun.kill()
        log(f"   tunnel attempt {i + 1}/{attempts}: no URL within {wait_s} s (cloudflared rc {tun.poll()}); retrying")
    return None


def tunnel_probe(u, api_key, wait_s=180):
    """Background: confirm the public URL answers (a fresh hostname can take a minute to resolve); if it never does,
    open a new tunnel once and announce the new URL."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            req = urllib.request.Request(f"{u}/health", headers={"Authorization": f"Bearer {api_key}"})
            urllib.request.urlopen(req, timeout=10).read()
            log(f"   tunnel reachable from outside: {u} ({time.time() - t0:.0f} s after it opened)")
            return
        except Exception:  # noqa: BLE001
            time.sleep(10)
    log(f"   tunnel URL {u} did not answer in {wait_s} s: opening a new one")
    if TUN[0] is not None:
        TUN[0].kill()
    new = start_tunnel()
    if new:
        STATE["url"] = new
        publish("tunnel-url", endpoint=new)
        log(f"#  NEW ENDPOINT: {new}   (the earlier URL never resolved; the API key is unchanged)")
    else:
        publish("tunnel-failed", note="server still reachable inside the kernel on :8000")


if CFG["tunnel"]:
    url = start_tunnel()
    if url:
        publish("tunnel-url", endpoint=url)
        threading.Thread(target=tunnel_probe, args=(url, API_KEY), daemon=True).start()
    else:
        publish("tunnel-failed", note="server still reachable inside the kernel on :8000")
STATE["url"] = url or f"http://127.0.0.1:{PORT}"

# ----------------------------------------------------------------------------- 6. ready, self-test, keep alive
banner(6, "Ready", "self-test, then serving")


def _post(path, body):
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers=hdr), timeout=900)
    return json.loads(r.read().decode())


try:
    r = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi in five words."}], "max_tokens": 48, "temperature": 0})
    log("   self-test openai:", json.dumps(r["choices"][0]["message"]["content"])[:120], r["usage"])
    tools = [{"name": "read_file", "description": "Read a text file from disk.",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}, "required": ["path"]}}]
    msgs = [{"role": "user", "content": "Use the read_file tool to read /etc/hostname (at most 5 lines)."}]
    r = _post("/v1/messages", {"model": MODEL, "max_tokens": 256, "messages": msgs, "tools": tools, "temperature": 0})
    log("   self-test anthropic:", json.dumps(r["content"])[:200], r["stop_reason"])
    outs = [None, None]

    def _one(i, prompt):
        outs[i] = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": 48, "temperature": 0})
    ths = [threading.Thread(target=_one, args=(0, "Count from one to ten in words.")), threading.Thread(target=_one, args=(1, "Name three primary colors."))]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    log("   self-test concurrent:", [o["choices"][0]["message"]["content"][:50] for o in outs], f"{time.time() - t0:.1f}s")
except Exception as e:  # noqa: BLE001
    import traceback; log("   self-test failed:", traceback.format_exc()[-800:])

endpoint = STATE["url"]
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {endpoint}   (OpenAI at /v1, Anthropic at /v1/messages)")
log(f"#  API KEY  : {API_KEY}")
log(f"#  MODEL    : {MODEL}   (context {CFG['max_len']}, up to {MAX_STREAMS} streams)")
log("#" * 70)
log("#  Claude Code:")
log(f"#    ANTHROPIC_BASE_URL={endpoint} ANTHROPIC_AUTH_TOKEN={API_KEY} ANTHROPIC_MODEL={MODEL} \\")
log(f"#    ANTHROPIC_SMALL_FAST_MODEL={MODEL} CLAUDE_CODE_MAX_CONTEXT_TOKENS={CFG['max_len']} claude")
log("#  OpenAI-compatible clients: base URL " + endpoint + "/v1, model " + MODEL)
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)


def report_to_gateway(reason):
    """Burst adoption seam: tell the control-plane gateway this session is live.
    Non-fatal; the token is the bearer and never logged."""
    if not (CFG["burst_gateway"] and endpoint and CFG["burst_instance"]):
        return False
    body = {
        "instance": CFG["burst_instance"],
        "url": endpoint,
        "credential_ref": CFG["burst_credential_ref"] or None,
        "model": CFG["burst_model"] or None,
    }
    req = urllib.request.Request(
        f"{CFG['burst_gateway'].rstrip('/')}/admin/endpoints",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {CFG['burst_gateway_token']}",
                 "User-Agent": "burst-kernel/0.1"})
    try:
        urllib.request.urlopen(req, timeout=10)
        log(f"   gateway told ({reason})")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"   (gateway report failed: {e})")
        return False


# Keep the bearer key in the local launcher state; ntfy is only a progress relay.
publish("ready", endpoint=endpoint, model=MODEL, max_model_len=CFG["max_len"],
        keepalive_min=CFG["keepalive_min"], startup_secs=int(time.time() - T0))
report_to_gateway("ready")

if globals().get("SERVE_FOREVER", True):
    t_serve = time.time()
    idle_deadline = CFG["idle_min"] * 60
    last_reqs = STATE["requests"]
    last_busy = time.time()
    while time.time() - t_serve < CFG["keepalive_min"] * 60:
        time.sleep(120)
        if SCHED.thread is not None and not SCHED.thread.is_alive():
            log("   the engine scheduler exited before the scheduled auto-shutdown (that only fires after "
                f"{CFG['keepalive_min']} min of serving) — its last error is in the log above")
            publish("stopped", reason="scheduler-exit")
            sys.exit(1)
        up = int((time.time() - t_serve) / 60)
        # active/pending misses a request that arrives and finishes between two polls; a
        # change in the completed-request counter marks the session busy anyway.
        busy = bool(SCHED.active or SCHED.pending) or STATE["requests"] != last_reqs
        last_reqs = STATE["requests"]
        if busy:
            last_busy = time.time()
        idle_now = int((time.time() - last_busy) / 60)
        if up % 10 < 2:
            publish("heartbeat", up_min=up, endpoint=endpoint, requests=STATE["requests"], tokens=STATE["tokens"],
                    busy=busy, idle_min=idle_now)
            report_to_gateway("heartbeat")
        if idle_deadline and (time.time() - last_busy) >= idle_deadline:
            served_min = int((time.time() - t_serve) / 60)
            publish("idle-shutdown", idle_min=idle_now, served_min=served_min)
            log(f"#  idle for {idle_now} min (idle_min={CFG['idle_min']}) — exiting "
                f"to free the slot after {served_min} min of serving")
            break
    publish("auto-shutdown", served_min=int((time.time() - t_serve) / 60))
    SCHED.stop(); srv.shutdown()
    sys.exit(0)
