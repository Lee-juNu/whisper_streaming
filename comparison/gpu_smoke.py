"""Single-model GPU smoke test (host orchestrator, stdlib only). One owned child at a time:
refuses while npc_whisper / another smoke container / a live ASR process is present, never
stops services, and cleans up only its own PID/container. Prints one JSON report."""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from asrbench import guards  # noqa: E402

IMAGE = "whisper_streaming-whisper:latest"
KOTOBA_MODEL = "local/models/kotoba-whisper-v2"
AUDIO = "local/audio/ja_smoke_12s.wav"
NEMO_RUNTIME = ROOT / "local" / "nemo-runtime"
NEMO_BIN = NEMO_RUNTIME / "nemo-speech-0.1.0-linux-x86_64-cuda" / "bin" / "nemo-speech"
NEMO_MODELS = ROOT / "local" / "models"
NEMO_MODEL = NEMO_MODELS / "nemotron" / "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
NEMO_DEVICE = "cuda:0"
NEMO_BACKEND = "backend=CUDA0"  # observed in nemo-speech stderr: "[asr] model=... head=rnnt backend=CUDA0"
GPU_INDEX = 0  # only GPU 0 is supported (docker device=0, native cuda:0, nvidia-smi index 0)
LIVE_CONTAINER = "npc_whisper"
SMOKE_PREFIX = "asrbench-smoke-"
TOL_MIB = 100
MARK = "ASRBENCH_SMOKE_JSON="

# Runs inside the container: cold load, one warmup, one measured transcription (generator consumed).
KOTOBA_SCRIPT = r'''
import json, sys, time
t0 = time.perf_counter()
from faster_whisper import WhisperModel
m = WhisperModel(sys.argv[1], device="cuda", compute_type="float16", local_files_only=True)
t1 = time.perf_counter()
def run():
    segs, _ = m.transcribe(sys.argv[2], language="ja", beam_size=5, condition_on_previous_text=False, vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    return "".join(s.text for s in segs).strip()
warm = run()
t2 = time.perf_counter()
text = run()
t3 = time.perf_counter()
dev = m.model.device
assert dev == "cuda", "engine.device=%r (no CPU fallback allowed)" % dev
assert text, "empty transcript"
print("ASRBENCH_SMOKE_JSON=" + json.dumps({"engine_device": dev, "compute_type": m.model.compute_type,
      "cold_load_ms": (t1 - t0) * 1e3, "warmup_ms": (t2 - t1) * 1e3, "recognition_ms": (t3 - t2) * 1e3,
      "warmup_transcript": warm, "transcript": text}))
'''


def kotoba_argv(name):
    return ["docker", "run", "--rm", "--name", name, "--label", "asrbench.smoke=1",
            "--gpus", f"device={GPU_INDEX}", "--network", "none", "-v", f"{ROOT}:/cmp:ro",
            "--entrypoint", "python3.11", IMAGE, "-c", KOTOBA_SCRIPT, "/cmp/" + KOTOBA_MODEL, "/cmp/" + AUDIO]


def nemotron_default_argv():
    return [str(NEMO_BIN), "transcribe", str(ROOT / AUDIO), "--model", str(NEMO_MODEL),
            "--device", NEMO_DEVICE, "--language", "ja-JP", "--format", "json", "--stream", "--no-batching",
            "--asr.streaming.rnnt_right_context", "1"]


def abs_path(p):
    """Relative paths are taken from ROOT (the child also runs with cwd=ROOT), never from the caller's cwd."""
    p = Path(p)
    return (p if p.is_absolute() else ROOT / p).resolve()


def flag_values(argv, name):
    vals = []
    for i, a in enumerate(argv):
        if a == name:
            vals.append(argv[i + 1] if i + 1 < len(argv) else None)
        elif a.startswith(name + "="):
            vals.append(a[len(name) + 1:])
    return vals


def validate_native_argv(cmd):
    """transcribe + exactly one --device cuda:0 + an existing local .gguf (no download) + --format json."""
    allowed = {"--model", "--device", "--language", "--format", "--stream", "--no-batching",
               "--asr.streaming.rnnt_right_context"}
    if any(a.startswith("-") and a.split("=", 1)[0] not in allowed for a in cmd[2:]):
        raise guards.GuardError("native smoke accepts only the documented single-model transcription flags")
    exe = abs_path(cmd[0])
    if NEMO_RUNTIME.resolve() not in exe.parents or not exe.exists():
        raise guards.GuardError(f"nemotron binary must exist under {NEMO_RUNTIME.relative_to(ROOT)}")
    if cmd[1:2] != ["transcribe"]:
        raise guards.GuardError("nemotron: first argument must be the `transcribe` subcommand")
    if flag_values(cmd, "--backend") or flag_values(cmd, "--device") != [NEMO_DEVICE]:
        raise guards.GuardError(f"nemotron: exactly one explicit `--device {NEMO_DEVICE}` required (GPU 0 only)")
    models = flag_values(cmd, "--model")
    if len(models) != 1 or not models[0]:
        raise guards.GuardError("nemotron: exactly one `--model <local .gguf>` required")
    model = abs_path(models[0])
    if model.suffix != ".gguf" or NEMO_MODELS.resolve() not in model.parents or not model.exists():
        raise guards.GuardError(f"nemotron: --model must be an existing .gguf under {NEMO_MODELS.relative_to(ROOT)}")
    if flag_values(cmd, "--format") != ["json"]:
        raise guards.GuardError("nemotron: exactly one `--format json` required")
    return [str(exe)] + cmd[1:]


def native_result(res):
    """Transcript only from parsed JSON `text`; help text, empty JSON or a non-CUDA0 backend never count."""
    backend_ok = NEMO_BACKEND in res["stderr"]
    try:
        doc = json.loads(res["stdout"])
    except ValueError:
        return {"backend_cuda0": backend_ok, "stdout_json": False}, ""
    text = doc.get("text") if isinstance(doc, dict) else None
    text = text.strip() if isinstance(text, str) else ""
    return {"backend_cuda0": backend_ok, "stdout_json": True}, text


def check_idle(runner, proc_root="/proc"):
    """Fail closed unless no live ASR service and no other benchmark container is visible."""
    rc, out, err = runner(["docker", "ps", "--format", "{{.Names}}"])
    if rc != 0:
        raise guards.GuardError(f"docker ps failed ({(err or '').strip() or rc}); cannot verify idle GPU")
    names = out.split()
    if any(LIVE_CONTAINER in n for n in names):
        raise guards.GuardError(f"{LIVE_CONTAINER} is running. Stop it yourself first; this tool never stops services.")
    if any(n.startswith(SMOKE_PREFIX) for n in names):
        raise guards.GuardError("another benchmark smoke container is active; one model at a time")
    hits = guards.scan_cmdlines(guards.ancestor_pids(os.getpid(), proc_root), proc_root=proc_root)
    if hits:
        raise guards.GuardError("live ASR-like process present: "
                                + ", ".join(f"pid={h['pid']} {h['name']}" for h in hits))


def vram_used(runner, gpu):
    return next(g["memory_used_mib"] for g in guards.query_gpus(runner) if g["index"] == gpu)


def stop_own(proc, runner, container):
    """Kill and reap only our own child PID / uniquely named container."""
    if container:
        runner(["docker", "kill", container])
    proc.kill()
    try:
        return proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        return "", ""


def run_child(argv, timeout, popen, runner, container=None):
    # Pin CUDA ordinal 0 to nvidia-smi index 0 so the monitored GPU is the one actually used.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    t0 = time.monotonic()
    proc = popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT), env=env)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        out, err = stop_own(proc, runner, container)
    except BaseException:  # KeyboardInterrupt or any error while the child is still running
        stop_own(proc, runner, container)
        raise
    return {"pid": proc.pid, "returncode": proc.returncode, "timed_out": timed_out,
            "subprocess_wall_ms": (time.monotonic() - t0) * 1e3, "stdout": out or "", "stderr": err or ""}


def container_gone(runner, name):
    rc, out, _ = runner(["docker", "ps", "-aq", "--filter", f"name=^/{name}$"])
    if rc == 0 and out.strip():
        runner(["docker", "rm", "-f", name])  # own container only (unique name)
        rc, out, _ = runner(["docker", "ps", "-aq", "--filter", f"name=^/{name}$"])
    return rc == 0 and not out.strip()


def wait_release(runner, gpu, before, wait_s, sleep, tol=TOL_MIB):
    deadline = time.monotonic() + wait_s
    while True:
        used = vram_used(runner, gpu)
        if used <= before + tol or time.monotonic() >= deadline:
            return used, used <= before + tol
        sleep(1)


def smoke(args, runner=guards.run_cmd, popen=subprocess.Popen, sleep=time.sleep, proc_root="/proc"):
    if not args.allow_gpu:
        raise guards.GuardError("GPU smoke needs explicit --allow-gpu")
    if args.gpu != GPU_INDEX:
        raise guards.GuardError(f"only GPU {GPU_INDEX} is supported (got --gpu {args.gpu})")
    container = None
    if args.model == "kotoba":
        for rel in (KOTOBA_MODEL, AUDIO):
            if not (ROOT / rel).exists():
                raise guards.GuardError(f"missing {rel}")
        container = SMOKE_PREFIX + uuid.uuid4().hex[:8]
        argv = kotoba_argv(container)
    else:
        argv = validate_native_argv(args.cmd or nemotron_default_argv())
    with guards.BenchLock(args.lock):
        check_idle(runner, proc_root)
        before = vram_used(runner, args.gpu)
        gone = {}
        try:
            res = run_child(argv, args.timeout, popen, runner, container)
        finally:
            if container:
                gone["container_removed"] = container_gone(runner, container)
        gone["process_reaped"] = res["returncode"] is not None and not Path(proc_root, str(res["pid"])).exists()
        after, released = wait_release(runner, args.gpu, before, args.release_wait, sleep)
    rep = {"model": args.model, "returncode": res["returncode"], "timed_out": res["timed_out"],
           "subprocess_wall_ms": res["subprocess_wall_ms"], "cleanup": gone,
           "vram": {"label": guards.VRAM_LABEL, "before_mib": before, "after_mib": after,
                    "tolerance_mib": TOL_MIB, "released": released},
           "stderr_tail": res["stderr"][-1500:]}
    native_ok = True
    if container:
        line = next((ln for ln in res["stdout"].splitlines() if ln.startswith(MARK)), None)
        rep["engine"] = json.loads(line[len(MARK):]) if line else None
        text = (rep["engine"] or {}).get("transcript", "")
    else:
        # Native CLI does not report phases separately; subprocess_wall_ms covers process start, model load,
        # recognition and exit, so it is NOT recognition latency.
        checks, text = native_result(res)
        native_ok = checks["backend_cuda0"] and checks["stdout_json"]
        rep["engine"] = {"cold_load_ms": "unknown", "warmup_ms": "unknown", "recognition_ms": "unknown",
                         "transcript": text, **checks}
        rep["stdout_tail"] = res["stdout"][-1500:]
    rep["transcript_nonempty"] = bool(text)
    rep["ok"] = (res["returncode"] == 0 and not res["timed_out"] and released
                 and all(gone.values()) and bool(text) and native_ok)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["kotoba", "nemotron"], required=True)
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--release-wait", type=float, default=60)
    ap.add_argument("--lock", default=str(guards.DEFAULT_LOCK))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.cmd[:1] == ["--"]:
        args.cmd = args.cmd[1:]
    try:
        rep = smoke(args)
    except guards.GuardError as e:
        print(json.dumps({"ok": False, "refused": str(e)}, ensure_ascii=False))
        return 2
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
