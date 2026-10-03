"""CLI: preflight | mock | run | score | server-command | postfilter-inspect.

Nothing here installs, downloads, starts or stops anything. Only `run --allow-inference`
loads a local model / talks to the dedicated loopback test server."""
import argparse
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

from . import ROOT, adapters, audio, dataset, guards, mock, postfilter, runner


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _fail(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


def cmd_preflight(a):
    rep = {"python": sys.version.split()[0], "python_3_12_plus": sys.version_info >= (3, 12), "errors": []}
    lock, sha = runner.load_lock()
    rep["lock_sha256"] = sha
    rep["optional_packages"] = {m: importlib.util.find_spec(m) is not None
                                for m in ("faster_whisper", "ctranslate2", "numpy", "onnxruntime")}
    rep["profiles"] = {}
    for p in sorted((ROOT / "profiles").glob("*.json")):
        prof, meta = runner.load_profile(p)
        try:
            pf = postfilter.load_postfilter(prof.get("postfilter"), ROOT).meta
        except (OSError, SyntaxError, postfilter.PostfilterError) as e:
            pf = {"status": "error", "error": str(e)}
        rep["profiles"][prof["name"]] = {
            "modes": prof["modes"], "sha256": meta["sha256"],
            "placeholders": guards.find_placeholders([prof.get("model"), prof.get("runtime")]),
            "postfilter": pf}
    if a.dataset:
        items, issues = dataset.load_dataset(a.dataset, allow_placeholders=a.allow_placeholders)
        rep["dataset"] = {"path": a.dataset, "items": len(items), "issues": issues}
        if dataset.has_errors(issues):
            rep["errors"].append("dataset has errors")
    if a.gpu:  # only on request: runs nvidia-smi queries (read-only)
        try:
            rep["gpu"] = {"gpus": guards.query_gpus(), "compute_apps": guards.query_compute_apps(),
                          "wsl": guards.is_wsl()}
        except guards.GuardError as e:
            rep["errors"].append(str(e))
    if a.probe_server:  # only on request: GET /version and /v1/models on the loopback test port
        prof, _ = runner.load_profile(a.probe_server)
        try:
            eng = adapters.NemotronEngine(prof, audio.RealClock())
            info = eng.server_info()
            eng.check_identity(info)
            rep["server"] = info
        except (adapters.ProtocolError, guards.GuardError) as e:
            rep["errors"].append(str(e))
    _print(rep)
    return 1 if rep["errors"] else 0


def _run_mock(out, profile_name, mode, items, ds_path, bias):
    prof, meta = runner.load_profile(profile_name)
    clock = audio.FakeClock()
    if prof["engine"] == "faster_whisper":
        eng = mock.MockWhisperEngine(clock)
    else:
        eng = mock.MockNemotronEngine(clock, server_endpointing=mode == "replay-native")
    return runner.run_benchmark(prof, meta, mode, items, eng, clock, out, bias=bias, dataset_path=ds_path)


def cmd_mock(a):
    root = Path(a.out_dir) / f"mock-{runner.run_dir_name('all', 'modes', a.bias, True)}"
    ds_path = mock.write_synthetic_dataset(root / "data")
    items, issues = dataset.load_dataset(ds_path)
    if dataset.has_errors(issues):
        return _fail(f"synthetic dataset invalid: {issues}")
    res = {}
    for name in ("whisper-small", "nemotron-rc1-160ms"):
        prof, _ = runner.load_profile(name)
        for mode in prof["modes"]:
            out, s = _run_mock(root / "runs", name, mode, items, ds_path, a.bias)
            res[f"{name}/{mode}"] = {"dir": str(out), "micro_cer": s["cer"]["micro_cer"],
                                     "final_latency_p50_ms": s["final_latency_ms"]["p50"],
                                     "failures": len(s["failures"])}
    _print({"label": runner.MOCK_LABEL, "root": str(root), "runs": res})
    return 0


def cmd_run(a):
    if not a.allow_inference:
        return _fail("real runs need --allow-inference (loads a local model / uses the GPU). Use `mock` otherwise.")
    prof, pmeta = runner.load_profile(a.profile)
    if a.mode not in prof["modes"]:
        return _fail(f"{prof['name']} supports {prof['modes']}, not {a.mode}")
    ph = guards.find_placeholders([prof.get("model"), prof.get("runtime")])
    if ph:
        return _fail(f"profile still has placeholders {ph}; set local model paths after manual download")
    items, issues = dataset.load_dataset(a.dataset)
    if dataset.has_errors(issues):
        _print(issues)
        return _fail("dataset has errors")
    lock, _ = runner.load_lock()
    nemo = prof["engine"] == "nemotron_server"
    if nemo and (not a.server_pid or not a.attestation):
        return _fail("nemotron runs need --server-pid and --attestation (see examples/attestation.example.json)")
    clock = audio.RealClock()
    try:
        with guards.BenchLock(a.lock_file):
            gpu_idx = prof.get("gpu_index", 0)
            gpus = guards.query_gpus()
            gpu = next((g for g in gpus if g["index"] == gpu_idx), None)
            if gpu is None:
                return _fail(f"GPU index {gpu_idx} not found: {gpus}")
            allowed = {os.getpid()} | ({a.server_pid} if a.server_pid else set())
            inter = guards.assess_interference(
                guards.query_compute_apps(), allowed, guards.is_wsl(), a.operator_attest_gpu_exclusive,
                guards.scan_cmdlines(guards.ancestor_pids(os.getpid()) | allowed),
                required_pids=(a.server_pid,) if nemo else ())
            if inter["blocked"]:
                return _fail("GPU not exclusive (nothing was started or stopped):\n- " + "\n- ".join(inter["blocks"]))
            extra = {"gpu": gpu, "interference": inter}
            if nemo:
                extra["server"] = guards.verify_server_pid(a.server_pid)
                att = json.loads(Path(a.attestation).read_text(encoding="utf-8"))
                errs = guards.check_attestation(att, prof, a.mode, a.bias, lock)
                if errs:
                    return _fail("attestation mismatch:\n- " + "\n- ".join(errs))
                engine = adapters.NemotronEngine(prof, clock)
                info = engine.server_info()
                engine.check_identity(info)
                extra.update(attestation=att, server_info=info)
            sampler = guards.VramSampler(gpu_idx)
            sampler.start()
            try:
                if not nemo:
                    pf = postfilter.load_postfilter(prof.get("postfilter"), ROOT)
                    engine = adapters.FasterWhisperEngine(prof, pf)

                def vram():
                    sampler.stop()
                    return sampler.summary(gpu["memory_used_mib"], "nvidia-smi memory.used at preflight, "
                                                                   "before this run loaded anything")
                out, s = runner.run_benchmark(prof, pmeta, a.mode, items, engine, clock, a.out_dir, bias=a.bias,
                                              extra_meta=extra, dataset_path=a.dataset, vram_summary=vram)
            finally:
                sampler.stop()
    except (guards.GuardError, adapters.EngineUnavailable, adapters.ProtocolError,
            postfilter.PostfilterError) as e:
        return _fail(str(e))
    _print({"dir": str(out), "summary": {k: s[k] for k in ("cer", "final_latency_ms", "processing_ms", "failures")}})
    return 0


def cmd_score(a):
    items = None
    if a.dataset:
        items, issues = dataset.load_dataset(a.dataset, check_audio=False, allow_placeholders=True)
        if dataset.has_errors(issues):
            _print(issues)
            return _fail("dataset has errors")
    path, s = runner.rescore(a.run_dir, items)
    _print({"score_file": str(path), "cer": s["cer"], "nickname": {k: s["nickname"][k] for k in
            ("mention_recall", "candidate_list_precision", "hits", "misses", "false_insertions")},
            "failures": s["failures"]})
    return 0


def cmd_server_command(a):
    prof, _ = runner.load_profile(a.profile)
    if prof["engine"] != "nemotron_server":
        return _fail("server-command is only for nemotron profiles")
    if a.mode not in prof["modes"]:
        return _fail(f"{prof['name']} supports {prof['modes']}")
    print("# PRINT ONLY. Start it yourself in a manual exclusive GPU window; this tool never starts it.")
    print(shlex.join(guards.server_command(prof, a.mode, a.binary)))
    return 0


def cmd_postfilter_inspect(a):
    _print(postfilter.inspect_source(a.source))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m asrbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("preflight", help="static checks; no model load / network unless requested")
    s.add_argument("--dataset")
    s.add_argument("--allow-placeholders", action="store_true")
    s.add_argument("--gpu", action="store_true", help="also run read-only nvidia-smi queries")
    s.add_argument("--probe-server", metavar="PROFILE", help="GET /version,/v1/models on the loopback test port")
    s.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("mock", help="synthetic data + mock engines + FakeClock (stdlib only)")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--bias", choices=("off", "on"), default="off")
    s.set_defaults(fn=cmd_mock)
    s = sub.add_parser("run", help="REAL inference (explicit opt-in)")
    s.add_argument("--profile", required=True)
    s.add_argument("--mode", required=True, choices=runner.MODES)
    s.add_argument("--dataset", required=True)
    s.add_argument("--bias", choices=("off", "on"), default="off")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--allow-inference", action="store_true")
    s.add_argument("--operator-attest-gpu-exclusive", action="store_true")
    s.add_argument("--server-pid", type=int)
    s.add_argument("--attestation")
    s.add_argument("--lock-file", default=str(guards.DEFAULT_LOCK))
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("score", help="re-score an existing run directory")
    s.add_argument("run_dir")
    s.add_argument("--dataset", help="use (corrected) truth from this dataset instead of the stored snapshot")
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("server-command", help="print the test-server command (never executes it)")
    s.add_argument("--profile", required=True)
    s.add_argument("--mode", required=True, choices=runner.MODES)
    s.add_argument("--binary")
    s.set_defaults(fn=cmd_server_command)
    s = sub.add_parser("postfilter-inspect", help="list extractable names in ws_server.py (ast only, no import)")
    s.add_argument("--source", default=str((ROOT / ".." / "ws_server.py").resolve()))
    s.set_defaults(fn=cmd_postfilter_inspect)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
