"""Public-sample GPU speed benchmark: whisper-small / kotoba-whisper-v2 (faster-whisper in the existing
docker image) vs Nemotron 3.5 Q8 (native nemo-speech server, file mode). Speed only - accuracy is NOT
assessed, no reference/CER, no replay/end-of-speech latency. Stdlib only; never starts/stops services.

  python3 public_speed.py prepare
  python3 public_speed.py run --allow-gpu --repeats 8 [--nemotron-context3]
  (whisper-worker is invoked by `run` inside the container)
"""
import argparse, datetime, json, math, os, signal, subprocess, sys, threading, time, urllib.request, uuid, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "local/audio/sample_ja_speech.wav"
SAMPLES_DIR = ROOT / "local/audio/public_speed"
IMAGE = "whisper_streaming-whisper:latest"
PORT = 18101
JST = datetime.timezone(datetime.timedelta(hours=9))
CLIPS = [("short_a", 10, 2.5), ("short_b", 30, 3), ("medium_a", 50, 6),
         ("medium_b", 80, 8), ("long_a", 110, 12), ("long_b", 160, 18)]
WHISPER = {"small": "whisper-small", "kotoba": "kotoba-whisper-v2"}
WHISPER_OPTS = dict(language="ja", beam_size=5, condition_on_previous_text=False, vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300}, word_timestamps=False,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], initial_prompt=None,
                    no_speech_threshold=0.6, log_prob_threshold=-1.0)  # upstream defaults, explicit
WORKER_TIMEOUT_S, HTTP_TIMEOUT_S, BUDGET_S, VRAM_SLACK_MIB, VRAM_WAIT_S = 300, 30, 20 * 60, 100, 30
perf = time.perf_counter


def now():
    return datetime.datetime.now(JST).isoformat(timespec="seconds")


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:  # persisted per trial: survives a crash mid-run
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plan(ids, repeats):
    """2 warmups on the longest clip, then `repeats` rounds each rotated by one (deterministic)."""
    longest = max(CLIPS, key=lambda c: c[2])[0]
    warm = [("warm", r, ids[(r + i) % len(ids)]) for r in range(repeats) for i in range(len(ids))]
    return [("warmup", i, longest) for i in range(2)] + warm


def pctl(vals, p):
    """Nearest-rank percentile."""
    v = sorted(vals)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)] if v else None


# --------------------------------------------------------------------------- prepare
def prepare(a):
    src_meta = json.loads((SRC.parent / "sample_ja_speech.source.json").read_text())
    src_sha = sha256(SRC)
    if src_sha != src_meta["sha256"]:
        sys.exit(f"source sha256 {src_sha} != recorded {src_meta['sha256']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an existing sample set
    with wave.open(str(SRC), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype())
        if fmt != (1, 2, 16000, "NONE"):
            sys.exit(f"source is {fmt}; need 16 kHz PCM16 mono (no resampling is done here)")
        total = w.getnframes()
        clips = []
        for cid, off, dur in CLIPS:
            s, n = int(off * 16000), int(dur * 16000)
            if s + n > total:
                sys.exit(f"{cid}: {off}+{dur}s exceeds source ({total / 16000:.1f}s)")
            w.setpos(s)
            data = w.readframes(n)
            p = out / f"{cid}.wav"
            with open(p, "xb") as f, wave.open(f, "wb") as o:
                o.setnchannels(1), o.setsampwidth(2), o.setframerate(16000), o.writeframes(data)
            clips.append({"id": cid, "file": p.name, "sha256": sha256(p), "offset_s": off,
                          "duration_s": len(data) / 32000, "frames": len(data) // 2})
    rev = src_meta["revision"]
    write_json(out / "samples.json", {
        "created_at": now(), "format": "PCM16 16 kHz mono WAV, sample-exact cuts, no processing",
        "source": {"path": str(SRC), "sha256": src_sha, "bytes": src_meta["bytes"], "url": src_meta["url"],
                   "revision": rev, "readme": f"https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0-ggml/"
                                             f"blob/{rev}/README.md"},
        "license": "UNVERIFIED: the audio's origin/license is not recorded locally; the hosting repo's model "
                   "license does not necessarily cover this sample. Do not redistribute the clips.",
        "reference_text": None, "note": "speed-only clips; no reference transcripts, no CER", "clips": clips})
    print(f"prepared {len(clips)} clips -> {out}")


def load_samples(d):
    m = json.loads((Path(d) / "samples.json").read_text())
    for c in m["clips"]:
        c["path"] = str(Path(d) / c["file"])
        if sha256(c["path"]) != c["sha256"]:
            raise SystemExit(f"{c['path']}: sha256 mismatch vs samples.json")
    return m


# --------------------------------------------------------------------------- whisper worker (in container)
def whisper_worker(a):
    out = Path(a.out)
    emit = lambda **k: print(json.dumps(k, ensure_ascii=False), flush=True)
    prof = json.loads((ROOT / "profiles" / f"{WHISPER[a.profile]}.json").read_text())
    model_dir = prof["model"]["local_model_dir"]
    t = perf()
    import faster_whisper, ctranslate2
    from faster_whisper import WhisperModel
    import_ms = (perf() - t) * 1000
    emit(event="imported", import_ms=import_ms)
    t = perf()
    model = WhisperModel(model_dir, device="cuda", device_index=0, compute_type="float16", local_files_only=True)
    init_ms = (perf() - t) * 1000
    assert model.model.device == "cuda", "GPU backend required; no CPU fallback"
    emit(event="model_init", init_ms=init_ms)
    vers = {}
    from importlib import metadata
    for pkg in ("faster-whisper", "ctranslate2", "av", "onnxruntime", "numpy", "tokenizers"):
        try:
            vers[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            vers[pkg] = None
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15) if a.smi else None
    meta = {"profile": a.profile, "model_dir": model_dir, "model_bin_sha256": sha256(Path(model_dir, "model.bin")),
            "model_revision": prof["model"]["revision"], "import_ms": import_ms, "init_ms": init_ms,
            **{k: getattr(model.model, k, None) for k in ("device", "device_index", "compute_type")},
            "cuda_device_count": ctranslate2.get_cuda_device_count(),
            "container_nvidia_smi": smi.stdout.strip() if smi and smi.returncode == 0 else None,
            "python": sys.version.split()[0], "versions": vers, "transcribe_options": WHISPER_OPTS,
            "postfilter": "none: raw ASR text, production post-filters NOT applied"}
    write_json(out / f"{a.profile}.worker.json", meta)
    clips = {c["id"]: c for c in json.loads((out / "samples.json").read_text())["clips"]}
    for phase, rnd, cid in json.loads((out / "plan.json").read_text()):
        c, rec = clips[cid], {"engine": a.profile, "phase": phase, "round": rnd, "sample": cid}
        try:
            t = perf()  # file decode + VAD + recognition; generator consumed inside the timer
            segs, info = model.transcribe(str(out / "samples" / c["file"]), **WHISPER_OPTS)
            text = "".join(s.text for s in segs).strip()
            rec["elapsed_ms"] = (perf() - t) * 1000
            rec.update(raw_text=text, language=info.language, decoded_duration_s=info.duration,
                       after_vad_duration_s=info.duration_after_vad, error=None if text else "empty_text")
            if abs(info.duration - c["duration_s"]) > 0.002:
                rec["error"] = "decoded duration mismatch"
        except Exception as e:  # recorded, never counted as success
            rec.update(raw_text=None, error=f"{type(e).__name__}: {e}"[:500])
        rec.update(duration_s=c["duration_s"], ok=rec["error"] is None,
                   rtf=rec.get("elapsed_ms", 0) / 1000 / c["duration_s"] if "elapsed_ms" in rec else None)
        append_jsonl(out / f"{a.profile}.trials.jsonl", rec)
        emit(event="trial", phase=phase, sample=cid, ok=rec["ok"], elapsed_ms=rec.get("elapsed_ms"))
    emit(event="model_complete", profile=a.profile)


# --------------------------------------------------------------------------- host side
def sh(argv, timeout=30):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def preflight(guards):
    rc, out, err = sh(["docker", "ps", "--filter", "name=npc_whisper", "--format", "{{.Names}}"])
    if rc != 0:
        raise guards.GuardError(f"docker ps failed ({err}); failing closed")
    if out:
        raise guards.GuardError(f"npc_whisper is running ({out}); the outer wrapper must stop it first")
    hits = guards.scan_cmdlines(guards.ancestor_pids(os.getpid()))
    if hits:
        raise guards.GuardError(f"live ASR processes present: {hits}")
    rc, others, err = sh(["docker", "ps", "--format", "{{.Names}}"] )
    if rc != 0 or any(n.startswith(("public-speed-", "asrbench-smoke-")) for n in others.splitlines()):
        raise guards.GuardError("another benchmark container exists or cannot inspect containers")
    if guards.listeners(PORT):
        raise guards.GuardError(f"127.0.0.1:{PORT} already has a listener")


def vram_used(guards):
    return guards.query_gpus()[0]["memory_used_mib"]


def wait_released(guards, baseline, log):
    t0 = time.monotonic()
    while time.monotonic() - t0 < VRAM_WAIT_S:
        used = vram_used(guards)
        if used <= baseline + VRAM_SLACK_MIB:
            log["vram_released"] = {"used_mib": used, "waited_s": round(time.monotonic() - t0, 1)}
            return
        time.sleep(1)
    raise guards.GuardError(f"VRAM not released within {VRAM_WAIT_S}s (used {used} > baseline {baseline}+"
                            f"{VRAM_SLACK_MIB}); stopping, nothing further is started")


def run_whisper(name, run, deadline, guards):
    cname = f"public-speed-{name}-{uuid.uuid4().hex[:8]}"
    argv = ["docker", "run", "--rm", "--pull", "never", "--name", cname, "--gpus", "device=0",
            "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{run}:{run}:rw", "--entrypoint", "python3.11", IMAGE,
            str(ROOT / "public_speed.py"), "whisper-worker", "--profile", name, "--out", str(run), "--smi"]
    info = {"argv": argv, "container": cname}
    timeout = min(WORKER_TIMEOUT_S, deadline - time.monotonic())
    with open(run / f"{name}.stdout.log", "w") as so, open(run / f"{name}.stderr.log", "w") as se:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=se, stdin=subprocess.DEVNULL, text=True)

        def pump():
            for line in proc.stdout:
                so.write(line), so.flush()
                if '"model_init"' in line or '"model_complete"' in line:
                    print(f"[{name}] {line.strip()}", flush=True)
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            info["returncode"] = proc.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            info["error"] = f"worker timeout {timeout:.0f}s"
        finally:
            if proc.poll() is None:
                sh(["docker", "kill", cname]), sh(["docker", "rm", "-f", cname])
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill(), proc.wait(5)
            th.join(10)
            rc, left, _ = sh(["docker", "ps", "-a", "--filter", f"name=^/{cname}$", "--format", "{{.ID}}"])
            if left:
                sh(["docker", "rm", "-f", cname])
                rc, left, _ = sh(["docker", "ps", "-a", "--filter", f"name=^/{cname}$", "--format", "{{.ID}}"])
            info["reaped"], info["container_removed"] = proc.returncode is not None, rc == 0 and not left
    if not info["container_removed"]:
        raise guards.GuardError(f"container {cname} could not be confirmed removed")
    if info.get("returncode") not in (0, None) and "error" not in info:
        info["error"] = f"worker exit {info['returncode']} (see {name}.stderr.log)"
    return info


def http(opener, url, data=None, ctype=None):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if ctype else {})
    with opener.open(req, timeout=HTTP_TIMEOUT_S) as r:
        return json.loads(r.read())


def multipart(fields, fname, data):
    b = "psb" + uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n'.encode() + data + b"\r\n" + f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def run_nemotron(name, rc_ctx, run, deadline, samples, steps):
    prof = json.loads((ROOT / "profiles" / "nemotron-rc1-160ms.json").read_text())
    binary, gguf = prof["runtime"]["binary"], prof["model"]["local_gguf_path"]
    argv = [binary, "serve", "--host", "127.0.0.1", "--port", str(PORT), "--no-warmup",
            "--asr.model.path", gguf, "--device", "cuda:0", "--no-ui", "--asr.batching.enabled=false",
            "--asr.endpointing.enable=false", "--asr.streaming.rnnt_right_context", str(rc_ctx)]
    base = f"http://127.0.0.1:{PORT}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    info = {"argv": argv, "rnnt_right_context": rc_ctx, "gguf_sha256": sha256(gguf), "binary_sha256": sha256(binary),
            "model_revision": prof["model"]["revision"], "runtime": prof["runtime"],
            "label": "native file mode (/v1/audio/transcriptions), NOT streaming replay; timed = read file + "
                     "multipart build + HTTP + full JSON, i.e. includes API overhead unlike the direct "
                     "faster-whisper Python call; not pure kernel time",
            "binary_version": " ".join(sh([binary, "--version"], 15)[1:])[:300]}
    trials = run / f"{name}.trials.jsonl"
    with open(run / f"{name}.server.stdout.log", "w") as so, open(run / f"{name}.server.stderr.log", "w") as se:
        t0 = perf()
        proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited early rc={proc.returncode}")
                if perf() - t0 > WORKER_TIMEOUT_S or time.monotonic() > deadline:
                    raise RuntimeError("server not ready within timeout/budget")
                try:
                    ready = http(opener, base + "/v1/models")
                    break
                except Exception:
                    time.sleep(0.2)
            info["startup_to_ready_ms"] = (perf() - t0) * 1000
            models = http(opener, base + "/v1/models")
            info.update(ready=ready, models=models)
            for k in ("version",):
                try:
                    info[k] = http(opener, base + "/" + k)
                except Exception as e:
                    info[k] = f"unavailable: {e}"
            ids = [m.get("id", "") for m in models.get("data", [])]
            mid = next((i for i in ids if "nemotron-3.5-asr-streaming" in i), None)
            if not mid or any("speech-streaming-en" in i for i in ids):
                raise RuntimeError(f"unexpected model inventory {ids}")
            if not any(m.get("id") == mid and m.get("device") == "cuda:0" for m in models.get("data", [])):
                raise RuntimeError("server model does not report exact device cuda:0")
            info["model_id"] = mid
            print(f"[{name}] ready in {info['startup_to_ready_ms']:.0f} ms, model {mid}", flush=True)
            fields = {"model": mid, "language": "ja-JP", "verbatim": "true", "response_format": "json"}
            for phase, rnd, cid in steps:
                if time.monotonic() > deadline:
                    info["error"] = "run budget exhausted"
                    break
                c, rec = samples[cid], {"engine": name, "phase": phase, "round": rnd, "sample": cid}
                try:
                    t = perf()
                    body, ctype = multipart(fields, c["file"], Path(c["path"]).read_bytes())
                    text = (http(opener, base + "/v1/audio/transcriptions", body, ctype).get("text") or "").strip()
                    rec["elapsed_ms"] = (perf() - t) * 1000
                    rec.update(raw_text=text, error=None if text else "empty_text")
                except Exception as e:
                    rec.update(raw_text=None, error=f"{type(e).__name__}: {e}"[:500])
                rec.update(duration_s=c["duration_s"], ok=rec["error"] is None,
                           rtf=rec["elapsed_ms"] / 1000 / c["duration_s"] if "elapsed_ms" in rec else None)
                append_jsonl(trials, rec)
            print(f"[{name}] model_complete", flush=True)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:500]
        finally:  # owned process only: SIGTERM group, then SIGKILL, always reaped
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL), proc.wait(10)
            info["reaped"], info["exit_code"] = proc.returncode is not None, proc.returncode
    return info


def summarize(run, engines):
    out = {"note": "speed only; accuracy NOT assessed (transcript differences are expected); no replay/"
                   "end-of-speech latency claims; percentiles are nearest-rank; errors excluded from stats",
           "engines": {}}
    for name, info in engines.items():
        p = run / f"{name}.trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        warm = [r for r in rows if r["phase"] == "warm"]
        ok = [r for r in warm if r["ok"]]
        stat = lambda rs, k: {"n": len(rs), "p50": pctl([r[k] for r in rs], 50), "p95": pctl([r[k] for r in rs], 95)}
        out["engines"][name] = {
            "cold": {k: info.get(k) for k in ("import_ms", "init_ms", "startup_to_ready_ms")},
            "warmup_ms": [r.get("elapsed_ms") for r in rows if r["phase"] == "warmup"],
            "warm_calls": len(warm), "errors": len(warm) - len(ok),
            "pooled": {"elapsed_ms": stat(ok, "elapsed_ms"), "rtf": stat(ok, "rtf")},
            "per_sample": {cid: {"elapsed_ms": stat(rs, "elapsed_ms"), "rtf": stat(rs, "rtf")}
                           for cid in dict.fromkeys(r["sample"] for r in ok)
                           for rs in [[r for r in ok if r["sample"] == cid]]},
            "vram": info.get("vram"), "error": info.get("error")}
    return out


def run_bench(a):
    sys.path.insert(0, str(ROOT))
    from asrbench import guards
    if not a.allow_gpu:
        sys.exit("refusing: pass --allow-gpu (real GPU run)")
    if not 1 <= a.repeats <= 20:
        sys.exit("repeats must be 1..20 for a bounded run")
    start = time.monotonic()
    deadline = start + BUDGET_S
    m = load_samples(a.samples)
    run = ROOT / "results" / "public_speed" / f"{datetime.datetime.now(JST):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    with guards.BenchLock() as lock:
        assert not os.get_inheritable(lock.f.fileno())  # docker/server children must not hold the lock
        preflight(guards)
        rc, image_id, err = sh(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
        if rc != 0:
            sys.exit(f"image {IMAGE} not present ({err}); nothing is pulled or built")
        run.mkdir(parents=True, exist_ok=False)
        (run / "samples").mkdir()
        for c in m["clips"]:
            (run / "samples" / c["file"]).write_bytes(Path(c["path"]).read_bytes())
        write_json(run / "samples.json", {k: v for k, v in m.items()})
        steps = plan([c["id"] for c in m["clips"]], a.repeats)
        write_json(run / "plan.json", steps)
        samples = {c["id"]: dict(c, path=str(run / "samples" / c["file"])) for c in m["clips"]}
        meta = {"started_at": now(), "host_gpus": guards.query_gpus(), "image": IMAGE, "image_id": image_id,
                "repeats": a.repeats, "budget_s": BUDGET_S, "gpu_index": 0, "engines": {},
                "implementation_sha256": sha256(__file__), "lock": json.loads((ROOT / "lock.json").read_text()),
                "wsl_compute_process_accounting_limited": guards.is_wsl(), "compute_apps_before": guards.query_compute_apps()}
        order = [("small", "w"), ("kotoba", "w"), ("nemotron_rc1", 1)] + ([("nemotron_rc3", 3)] if a.nemotron_context3 else [])
        try:
            for name, kind in order:
                if time.monotonic() > deadline:
                    meta["engines"][name] = {"error": "skipped: run budget exhausted"}
                    continue
                preflight(guards)
                load_before = list(os.getloadavg())
                baseline = vram_used(guards)
                sampler = guards.VramSampler(0)
                sampler.start()
                try:
                    if kind == "w":
                        info = run_whisper(name, run, deadline, guards)
                        wj = run / f"{name}.worker.json"
                        if wj.exists():
                            info.update(json.loads(wj.read_text()))
                    else:
                        info = run_nemotron(name, kind, run, deadline, samples, steps)
                finally:
                    sampler.stop()
                info["vram"] = sampler.summary(baseline, "nvidia-smi before engine start")
                info["host_loadavg_before"] = load_before
                info["host_loadavg_after"] = list(os.getloadavg())
                meta["engines"][name] = info
                if not info.get("reaped", True):
                    raise guards.GuardError(f"{name}: process not reaped")
                wait_released(guards, baseline, info)  # rc3 starts only after rc1 fully closed + released
        except guards.GuardError as e:
            meta["stopped"] = str(e)
            print(f"STOPPED: {e}", file=sys.stderr)
        finally:
            meta["finished_at"], meta["wall_s"] = now(), round(time.monotonic() - start, 1)
            write_json(run / "run.json", meta)
            summary = summarize(run, meta["engines"])
            write_json(run / "summary.json", summary)
    for name, s in summary["engines"].items():
        e, r = s["pooled"]["elapsed_ms"], s["pooled"]["rtf"]
        print(f"{name:14s} n={e['n']}/{s['warm_calls']} err={s['errors']} p50={e['p50']} ms p95={e['p95']} ms "
              f"rtf_p50={r['p50']} cold={s['cold']}")
    print(f"results: {run}")
    failed = meta.get("stopped") or any(s["error"] or s["errors"] or s["warm_calls"] != 6 * a.repeats
                                          for s in summary["engines"].values())
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", default=str(SAMPLES_DIR))
    w = sub.add_parser("whisper-worker")
    w.add_argument("--profile", choices=sorted(WHISPER), required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--smi", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--allow-gpu", action="store_true")
    r.add_argument("--repeats", type=int, default=8)
    r.add_argument("--samples", default=str(SAMPLES_DIR))
    r.add_argument("--nemotron-context3", action="store_true", help="optional second config, native file context")
    a = ap.parse_args(argv)
    return {"prepare": prepare, "whisper-worker": whisper_worker, "run": run_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
