#!/bin/bash
# Start the server on the VM: the published kernel unchanged, the engine from this checkout, no Cloudflare tunnel (you
# reach it through an SSH port forward), the compile cache in /kaggle/working. Runs in the tmux session "glm".
#   API_KEY=... STREAMS=3 PORT=8000 ./run.sh      (all optional; a key is generated when none is given)
set -euo pipefail
umask 077
export PATH="$HOME/.local/bin:$PATH"; source /mnt/data/envs/glm/bin/activate
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=/mnt/data/kaggle/working; mkdir -p "$RUN" /mnt/data/logs; cd "$RUN"
API_KEY=${API_KEY:-glm-$(openssl rand -hex 12)}; PORT=${PORT:-8000}
cat > serve_config.json <<JSON
{"tunnel": false, "skip_runtime": true, "api_key": "$API_KEY", "bind": "127.0.0.1", "keepalive_min": ${KEEPALIVE_MIN:-100000}, "streams": ${STREAMS:-3}, "port": $PORT}
JSON
chmod 600 serve_config.json
LOG=/mnt/data/logs/serve_$(date -u +%Y%m%d_%H%M).log
tmux kill-session -t glm 2>/dev/null || true
tmux new-session -d -s glm "PYTHONPATH=$REPO/glm53-flash/engine python $REPO/glm53-flash/kernel/serve_glm53.py 2>&1 | tee $LOG"
echo "server starting in tmux session 'glm' (tmux attach -t glm; or tail -f $LOG). READY in a few minutes when the exported and compiled programs match this chip, ~25-40 min when every program is traced and compiled."
echo "API key: $API_KEY"
echo "from your laptop:  gcloud compute tpus tpu-vm ssh <vm> --zone=<zone> -- -N -L $PORT:localhost:$PORT   then http://localhost:$PORT/v1 (OpenAI) or /v1/messages (Anthropic)"
