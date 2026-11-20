"""Make the phone bundle: chat-template prefix/suffix (so the phone needs no transformers), think tags,
and the controller weights. Run on the cluster/desktop, copy the output dir to the phone.

  python phone/export_bundle.py configs/eval_vibethinker.yaml models/controller_vibethinker phone_bundle/
"""
import json
import shutil
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from smallcoder.gen import ModelSpec, open_thinking

cfg_path, ctrl_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
spec = ModelSpec(**yaml.safe_load(open(cfg_path))["model"])
tok = AutoTokenizer.from_pretrained(spec.tokenizer or spec.path)
MARK = "@@USER_CONTENT@@"
rendered = open_thinking(tok.apply_chat_template([{"role": "user", "content": MARK}], add_generation_prompt=True,
                                                 tokenize=False), spec)
prefix, suffix = rendered.split(MARK)
bos = tok.bos_token or ""
out.mkdir(parents=True, exist_ok=True)
bundle = {"prefix": prefix[len(bos):] if bos and prefix.startswith(bos) else prefix,  # llama.cpp adds BOS
          "suffix_opened": suffix, "think_start": spec.think_start, "think_end": spec.think_end,
          "force_text": spec.force_text, "thinking": spec.thinking}
(out / "bundle.json").write_text(json.dumps(bundle, indent=1))
shutil.copy(ctrl_dir / "controller.npz", out / "controller.npz")
print("wrote", out, "| prompt suffix:", repr(suffix))
