import json, re, subprocess, sys, time, urllib.request
from pathlib import Path

TOPIC = "ktl-6418437426ec4b7baf23"
KEY = "1e0afcc97b0ba77076ef35a63664d578"
ALIAS = "Huihui-Qwen3.8-27B-abliterated"
REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
MODEL_FILE = "Huihui-Qwen3.8-27B-abliterated-UD-Q2_K_XL.gguf"

def ntfy(msg):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://ntfy.sh/{TOPIC}", data=msg.encode(), method="POST"), timeout=10)
    except Exception:
        pass

def say(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
    ntfy(f"p100-serve: {msg}")

say("step 1/5 gpu detect")
r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                    "--format=csv,noheader"], capture_output=True, text=True)
say("gpu: " + (r.stdout.strip().replace("\n", " | ") or r.stderr.strip()[:100]))
CC = r.stdout.split(",")[1].strip().replace(".", "") if "," in r.stdout else "60"

say(f"step 2/5 build llama.cpp (sm_{CC}, with CUDA driver stub fix)")
build = f"""
set -e
mkdir -p /kaggle/tmp/models /kaggle/tmp/logs
CUDA_ROOT=$(dirname "$(dirname "$(which nvcc)")")
echo "CUDA_ROOT=$CUDA_ROOT"
# Kaggle's image has no libcuda.so* anywhere (not even toolkit stubs), which breaks
# cmake's CUDA::cuda_driver imported target. GGML_CUDA_NO_VMM removes that link:
# VMM (cuMemCreate) only benefits modern GPUs; on Pascal we lose nothing.
rm -rf /kaggle/tmp/llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/tmp/llama.cpp 2>&1 | tail -1
cmake -S /kaggle/tmp/llama.cpp -B /kaggle/tmp/llama.cpp/build \
  -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES={CC} \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > /kaggle/tmp/logs/cmake.log 2>&1
cmake --build /kaggle/tmp/llama.cpp/build --config Release -j$(nproc) \
  > /kaggle/tmp/logs/build.log 2>&1
/kaggle/tmp/llama.cpp/build/bin/llama-server --version | head -1
"""
p = subprocess.run(["bash", "-lc", build], capture_output=True, text=True)
sys.stdout.write(p.stdout[-1500:]); sys.stdout.flush()
if p.returncode != 0:
    say(f"BUILD FAILED rc={p.returncode}; tail of logs:")
    for lg in ("cmake.log", "build.log"):
        f = Path("/kaggle/tmp/logs") / lg
        if f.exists():
            sys.stdout.write(f"--- {lg} ---\n" + "\n".join(
                f.read_text(errors="replace").splitlines()[-25:]) + "\n"); sys.stdout.flush()
    say("build failed - kernel exits")
    sys.exit(1)
say("build OK")

say("step 3/5 download GGUF (~10 GB)")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, allow_patterns=[MODEL_FILE],
                      local_dir="/kaggle/tmp/models")
    size = (Path("/kaggle/tmp/models") / MODEL_FILE).stat().st_size
    say(f"download OK ({size/1e9:.1f} GB)")
except Exception as ex:
    say("download failed: " + str(ex)[-300:])
    sys.exit(1)

say("step 4/5 launch llama-server")
start = """
source /dev/stdin <<'CFG'
KEY=1e0afcc97b0ba77076ef35a63664d578
ALIAS=Huihui-Qwen3.8-27B-abliterated
MODEL_FILE=Huihui-Qwen3.8-27B-abliterated-UD-Q2_K_XL.gguf
CFG
pkill -x llama-server 2>/dev/null; sleep 2
BIN=/kaggle/tmp/llama.cpp/build/bin/llama-server
# 128k context with q4_0 KV (16 full-attn layers -> ~2 GB KV at 128k). The ngl/FA
# ladder backs off if VRAM allocation fails at load time.
for FLAGS in "-c 128000 -fa on --cache-type-k q4_0 --cache-type-v q4_0" \
             "-c 128000 -fa on --cache-type-k q8_0 --cache-type-v q8_0" \
             "-c 128000 -fa off"; do
  for NGL in 99 90 80 60; do
    echo "trying: $FLAGS -ngl $NGL"
    nohup $BIN -m "/kaggle/tmp/models/$MODEL_FILE" -a "$ALIAS" \
      --host 127.0.0.1 --port 8080 -ngl $NGL $FLAGS \
      --cache-reuse 256 --jinja --no-webui --threads 4 \
      > /kaggle/tmp/logs/server.log 2>&1 &
    PID=$!
    OK=0
    for i in $(seq 1 75); do
      sleep 2
      curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && OK=1 && break
      kill -0 $PID 2>/dev/null || break
    done
    if [ "$OK" = "1" ]; then echo "UP: $FLAGS -ngl $NGL"; exit 0; fi
    pkill -x llama-server 2>/dev/null; sleep 2
  done
done
echo "SERVER FAILED"; tail -30 /kaggle/tmp/logs/server.log; exit 1
"""
p = subprocess.run(["bash", "-lc", start], capture_output=True, text=True)
sys.stdout.write(p.stdout[-1200:]); sys.stdout.flush()
if p.returncode != 0:
    say("server failed to start"); sys.exit(1)
say("server UP on 127.0.0.1:8080")

say("step 5/5 tunnel")
cf = """
set -e
if [ ! -x /kaggle/tmp/cloudflared ]; then
  echo "downloading cloudflared..."
  curl -sL --retry 3 -o /kaggle/tmp/cloudflared \\
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /kaggle/tmp/cloudflared
fi
/kaggle/tmp/cloudflared --version 2>&1 | head -1
"""
p = subprocess.run(["bash", "-lc", cf], capture_output=True, text=True)
sys.stdout.write((p.stdout + p.stderr)[-400:]); sys.stdout.flush()
if p.returncode != 0:
    say("cloudflared download failed")
    sys.exit(1)

URL = ""
for attempt in range(3):
    subprocess.run(["bash", "-lc",
        "pkill -x cloudflared 2>/dev/null; sleep 1; "
        "nohup /kaggle/tmp/cloudflared tunnel --url http://127.0.0.1:8080 --no-autoupdate "
        "> /kaggle/tmp/logs/tunnel.log 2>&1 &"], capture_output=True, text=True)
    for _ in range(45):
        time.sleep(2)
        try:
            log = Path("/kaggle/tmp/logs/tunnel.log").read_text(errors="replace")
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if m:
                URL = m.group(0)
                break
        except Exception:
            pass
    if URL:
        break
    sys.stdout.write(f"--- tunnel attempt {attempt + 1} failed; tunnel.log tail ---\n")
    sys.stdout.write("\n".join(Path("/kaggle/tmp/logs/tunnel.log")
                               .read_text(errors="replace").splitlines()[-12:]) + "\n")
    sys.stdout.flush()
    say(f"tunnel attempt {attempt + 1}/3 failed - retrying")
if not URL:
    say("tunnel failed after 3 attempts")
    sys.exit(1)

say(f"READY endpoint={URL}/v1 api_key={KEY} model={ALIAS}")
test = subprocess.run(
    ["bash", "-lc",
     "curl -s http://127.0.0.1:8080/v1/chat/completions "
     "-H 'Content-Type: application/json' "
     f"-H 'Authorization: Bearer {KEY}' "
     "-d '{\"model\":\"" + ALIAS + "\",\"messages\":[{\"role\":\"user\","
     "\"content\":\"Say OK\"}],\"max_tokens\":8}'"],
    capture_output=True, text=True).stdout
try:
    reply = json.loads(test)["choices"][0]["message"]["content"]
except Exception:
    reply = "(no reply: " + test[:120] + ")"
say("self-test reply: " + reply)

i = 0
while True:
    time.sleep(60); i += 1
    ok = subprocess.run(["bash", "-lc", "curl -s --max-time 5 http://127.0.0.1:8080/health"],
                        capture_output=True, text=True).stdout
    if "ok" not in ok:
        say("health check failed - restarting server")
        subprocess.run(["bash", "-lc", start], capture_output=True, text=True)
        say("restart complete")
    if i % 15 == 0:
        say(f"keepalive {URL}/v1")
