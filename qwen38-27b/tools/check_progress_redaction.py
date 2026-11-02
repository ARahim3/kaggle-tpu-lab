#!/usr/bin/env python3
"""Check the Qwen progress-event redaction without importing TPU dependencies."""
import ast
import json
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "kernel" / "serve_qwen38.py"
source_text = SOURCE.read_text()
tree = ast.parse(source_text)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
sanitizer = functions["sanitize_progress"]
publisher = functions["publish"]

sanitizer_source = ast.get_source_segment(source_text, sanitizer)
assert '"[REDACTED]"' in sanitizer_source
assert "casefold" in sanitizer_source and "replace" in sanitizer_source and "endswith" in sanitizer_source
assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == "sanitize_progress" for node in ast.walk(sanitizer))

publisher_source = ast.get_source_segment(source_text, publisher)
assert "safe_extra = sanitize_progress(extra)" in publisher_source
assert "json.dumps(safe_extra)" in publisher_source
assert 'json.dumps({"phase": phase, **safe_extra})' in publisher_source

# The notebook embeds the kernel as JSON-escaped source lines. Check the changed
# helper/publisher lines without decoding or executing notebook content.
notebook = SOURCE.parents[1] / "notebook" / "qwen38-tpu-serve.ipynb"
notebook_text = notebook.read_text()
for node in (sanitizer, publisher):
    segment = ast.get_source_segment(source_text, node)
    for line in segment.splitlines(keepends=True):
        notebook_line = line if line.endswith("\n") else line + "\n"
        assert json.dumps(notebook_line, ensure_ascii=False) in notebook_text, "kernel/notebook redaction drift"

print("Qwen progress-event redaction + notebook parity: PASS")
