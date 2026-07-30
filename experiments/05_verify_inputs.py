#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_inputs.py -- record SHA-256 of every input this audit consumes.

The audit's conclusions are only as fixed as the bytes they were computed
from.  This writes results/input_manifest.json, listing a digest for each
frame, script and report file read from the authors' deposit, so that any
later change to the deposit is detectable and every reported number can be
tied to a specific input state.
"""

from __future__ import annotations

import hashlib
import json
import os

import _path  # noqa: F401  (puts src/ on sys.path)
import nullius as N
TARGETS = [
    ("frame_data", os.path.join(N.DEPOSIT_DIR, "shared_raw_data",
                                "result4_frame_data", "exp_complex_B_semantic")),
    ("result4_scripts", os.path.join(N.DEPOSIT_DIR, "04_result4_paper", "scripts")),
    ("result5_scripts", os.path.join(N.DEPOSIT_DIR, "05_result4_refine_paper", "scripts")),
]
TOP_LEVEL = ["README.md", "MANIFEST.md", "SM_SCOPE_AND_RELATION_TO_MANUSCRIPT_SI.md",
             "DATA_AVAILABILITY_DRAFT.md", "SM_Reproducibility_Guide.pdf"]


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest = {"deposit_root": os.path.basename(N.DEPOSIT_DIR), "groups": {}}

    for name, root in TARGETS:
        if not os.path.isdir(root):
            manifest["groups"][name] = {"error": "missing", "path": root}
            continue
        entries, total = {}, 0
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, N.DEPOSIT_DIR).replace("\\", "/")
                entries[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
                total += os.path.getsize(p)
        manifest["groups"][name] = {"n_files": len(entries), "total_bytes": total,
                                    "files": entries}
        print(f"  {name:18s} {len(entries):5d} files  {total/1e6:9.1f} MB")

    top = {}
    for f in TOP_LEVEL:
        p = os.path.join(N.DEPOSIT_DIR, f)
        if os.path.exists(p):
            top[f] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    manifest["groups"]["deposit_documents"] = {"n_files": len(top), "files": top}
    print(f"  {'deposit_documents':18s} {len(top):5d} files")

    # a single digest over all per-file digests, for one-line citation
    allhashes = []
    for g in manifest["groups"].values():
        for rel, meta in sorted(g.get("files", {}).items()):
            allhashes.append(f"{rel}:{meta['sha256']}")
    rollup = hashlib.sha256("\n".join(sorted(allhashes)).encode()).hexdigest()
    manifest["rollup_sha256"] = rollup
    print(f"\n  rollup SHA-256 of all consumed inputs:\n  {rollup}")

    out = os.path.join(N.RESULTS_DIR, "input_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
