#!/usr/bin/env python3
"""Re-embed patches/mtp-rollback-v0280.diff into kernel/serve_qwen38.py.

Run after editing the patch file:  python tools/embed_patch.py
"""
import base64
import gzip
import re
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
diff = (repo / "patches" / "mtp-rollback-v0280.diff").read_bytes()
blob = base64.b64encode(gzip.compress(diff, 9)).decode()

script_path = repo / "kernel" / "serve_qwen38.py"
src = script_path.read_text()
new, n = re.subn(r'^MTP_PATCH_B64 = .*$',
                 f'MTP_PATCH_B64 = "{blob}"  # __EMBEDDED_PATCH__',
                 src, count=1, flags=re.M)
if n != 1:
    raise SystemExit("marker line MTP_PATCH_B64 = ... not found")
script_path.write_text(new)
print(f"embedded {len(diff)} bytes of diff as {len(blob)} chars of base64")
