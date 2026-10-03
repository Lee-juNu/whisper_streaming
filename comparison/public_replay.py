"""Bounded real-time paced replay on the public clips: latency only, accuracy NOT assessed (no reference/CER).
Five configs, run sequentially, one ASR model at a time (GPU 0):

  small / kotoba      faster-whisper in the existing docker image, replay-controlled (host endpointer 1500 ms)
  nemotron_ctl_rc1    native server, server endpointing OFF, same host endpointer decides the commit
  nemotron_nat_rc1/3  native server endpointing ON (token-silence 800 ms, vad_based=false), commit only at EOF

Each case = original clip + 3000 ms appended zeros, packets of 20 ms paced by the real clock.
clip_end_to_final_ms = receive time of the relevant final - (replay origin + clip duration). The anchor is the
ARTIFICIAL abrupt cut at the clip end (final 100 ms still voiced) followed by digital silence, NOT an annotated
natural end of sentence. Controlled = engine comparison; native 800 is a different endpointer (separate table).

  python3 public_replay.py run --allow-gpu
  (whisper-worker is invoked by `run` inside the container)
"""
import argparse, collections, copy, datetime, json, os, re, signal, subprocess, sys, time, uuid
from pathlib import Path

import public_speed as ps

ROOT = ps.ROOT
sys.path.insert(0, str(ROOT))
from asrbench import TEST_HOST, adapters, audio, guards, postfilter, runner  # noqa: E402  (stdlib only)

CLIP_IDS, WARMUP_CLIP, REPEATS, TAIL_MS = ("short_a", "medium_b", "long_b"), "short_a", 3, 3000
CONFIGS = {  # name: (profile, mode, server endpointing; None = faster-whisper worker)
    "small": ("whisper-small", "replay-controlled", None),
    "kotoba": ("kotoba-whisper-v2", "replay-controlled", None),
    "nemotron_ctl_rc1": ("nemotron-rc1-160ms", "replay-controlled", False),
    "nemotron_nat_rc1": ("nemotron-rc1-160ms", "replay-native", True),
    "nemotron_nat_rc3": ("nemotron-rc3-320ms", "replay-native", True),
}
BUDGET_S, WORKER_TIMEOUT_S, MIN_LEFT_S = 20 * 60, 240, 150
EXPECTED = REPEATS * len(CLIP_IDS)
ANCHOR = ("clip_end_to_final_ms: anchor = exact controlled clip end (abrupt cut, final 100 ms peak > 0.02) followed "
          "by 3000 ms appended zeros; NOT a manually annotated natural end of sentence. Includes decode/processing. "
          "Controlled: last non-empty final after the host-endpointer commit (clip end + 1500 ms). Native: last "
          "non-empty automatic final BEFORE the EOF commit; none => no_native_final (EOF commit never substituted).")


def plan():
    """1 short warmup (excluded from stats), then REPEATS rounds of all clips, each rotated by one."""
    return [("warmup", 0, WARMUP_CLIP)] + [("measured", r, CLIP_IDS[(r + i) % len(CLIP_IDS)])
                                          for r in range(REPEATS) for i in range(len(CLIP_IDS))]


def load_cases(d=ps.SAMPLES_DIR):
    cases = {}
    for c in ps.load_samples(d)["clips"]:  # sha256-verified against samples.json
        if c["id"] not in CLIP_IDS:
            continue
        clip = audio.read_wav(c["path"])
        tail = audio.peak(clip[-audio.ms_to_samples(100):])
        if tail <= 0.02:
            raise SystemExit(f"{c['id']}: final 100 ms peak {tail:.3f} <= 0.02, clip end is not an abrupt voiced cut")
        cases[c["id"]] = {"id": c["id"], "path": c["path"], "sha256": c["sha256"], "final_100ms_peak": tail,
                          "clip_ms": audio.samples_to_ms(len(clip)), "samples": clip + audio.silence(TAIL_MS)}
    if set(CLIP_IDS) - set(cases):
        raise SystemExit(f"missing clips {sorted(set(CLIP_IDS) - set(cases))} in {d}")
    return cases


def timing(rec, t0, clip_ms, mode):
    end = None if t0 is None else t0 + clip_ms / 1000
    for f in rec["finals"]:
        f["t_rel_ms"] = None if t0 is None else (f["t"] - t0) * 1000
        f["clip_end_to_final_ms"] = None if end is None else (f["t"] - end) * 1000
    content = [f for f in rec["finals"] if not f["empty"]]
    auto = [f for f in content if not f["after_commit"]]
    post = [f for f in content if f["after_commit"]]
    native = mode == "replay-native"
    pick = auto if native else post
    lat = pick[-1]["clip_end_to_final_ms"] if pick else None
    status = ("no_native_final" if native else "no_final") if lat is None else "early_final" if lat < 0 else "final"
    if native and auto and post and status == "final":
        status = "content_remaining_at_commit"
    return {"clip_end_to_final_ms": lat, "status": status, "n_finals": len(rec["finals"]),
            "auto_final_ms": [f["clip_end_to_final_ms"] for f in auto],
            "post_commit_final_ms": [f["clip_end_to_final_ms"] for f in post],
            "multiple_auto_finals": len(auto) > 1,
            "any_early_final": any(v is not None and v < 0 for v in (f["clip_end_to_final_ms"] for f in content)),
            "anomaly": "content final before commit with server endpointing off" if auto and not native else None}


def run_trial(name, engine, prof, mode, case, clock, phase, rnd):
    uid = f"{name}-{phase}{rnd}-{case['id']}-{uuid.uuid4().hex[:6]}"  # fresh session, no prompt carry
    inp = adapters.RecognitionInput(uid, uid, Path(case["path"]), case["sha256"], case["samples"], ())
    out = {"config": name, "mode": mode, "phase": phase, "round": rnd, "sample": case["id"], "utt_id": uid,
           "clip_ms": case["clip_ms"], "padded_ms": audio.samples_to_ms(len(case["samples"])), "error": None}
    events, t0 = [], None
    try:
        rec, events, t0 = runner.recognize(engine, prof, mode, None, inp, clock, [], "")
    except Exception as e:  # recorded, never counted as measured
        rec, events = {"finals": [], "failure": None}, getattr(e, "events", [])
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    out.update(rec)
    out.update(timing(rec, t0, case["clip_ms"], mode))
    rp = rec.get("replay") or {}
    if out["error"] is None:
        if out.get("anomaly"):
            out["error"] = out["anomaly"]
        elif rec.get("failure"):
            out["error"] = rec["failure"]
        elif rp.get("finish_ok") is False:
            out["error"] = "input_audio_buffer.committed not received after commit"
        elif mode == "replay-controlled" and rec.get("endpoint_decision") != "single_segment":
            out["error"] = f"host endpointer: {rec.get('endpoint_decision')}"
        elif mode == "replay-controlled":
            seg, sil = rec["segments"]["kept"][0], prof["endpoint"]["trailing_silence_ms"]
            if abs(seg["voice_end_ms"] - case["clip_ms"]) > 0.5 or abs(seg["endpoint_ms"] - case["clip_ms"] - sil) > 0.5:
                out["error"] = f"host endpointer segment {seg} != clip end / clip end + {sil} ms"
    partials = [e["t"] for e in events if e["type"] == adapters.EV_DELTA and str(e["raw"].get("delta", "")).strip()]
    out["first_nonempty_partial_from_start_ms"] = (partials[0] - t0) * 1000 if partials and t0 is not None else None
    out.update(measured=out["error"] is None and out["status"] == "final", replay_origin_t=t0,
               max_send_lag_ms=rp.get("max_send_lag_ms"))
    return out, [{"utt_id": uid, "i": i, "t": e["t"], "t_rel_ms": None if t0 is None else (e["t"] - t0) * 1000,
                  "type": e["type"], "raw": e["raw"]} for i, e in enumerate(events)]


def write_lines(path, objs):
    with open(path, "a", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_trials(name, engine, prof, mode, cases, clock, run, deadline=None):
    for phase, rnd, cid in plan():
        if deadline and time.monotonic() > deadline:
            raise RuntimeError("run budget exhausted")
        rec, events = run_trial(name, engine, prof, mode, cases[cid], clock, phase, rnd)
        write_lines(run / f"{name}.events.jsonl", events)
        write_lines(run / f"{name}.trials.jsonl", [rec])
        print(f"[{name}] {phase} {cid} {rec['status']} {rec['clip_end_to_final_ms']} err={rec['error']}", flush=True)


# --------------------------------------------------------------------------- whisper worker (in container)
def whisper_worker(a):
    run, (prof_name, mode, _) = Path(a.out), CONFIGS[a.config]
    prof, pmeta = runner.load_profile(prof_name)
    prof, clock = copy.deepcopy(prof), audio.RealClock()
    cases = load_cases()
    pf = postfilter.load_postfilter(prof.get("postfilter"), ROOT)  # filters empty -> nothing imported
    engine = adapters.FasterWhisperEngine(prof, pf)
    ct2 = engine.model.model
    if getattr(ct2, "device", None) != "cuda":
        raise SystemExit(f"GPU backend required, got {getattr(ct2, 'device', None)}; no CPU fallback")
    ps.write_json(run / f"{a.config}.worker.json", {
        "profile_file": pmeta, "engine": engine.describe(), "cold_init_ms": engine.load_ms, "clock": clock.kind,
        "ct2_device": ct2.device, "ct2_device_index": getattr(ct2, "device_index", None),
        "python": sys.version.split()[0]})
    run_trials(a.config, engine, prof, mode, cases, clock, run)


# --------------------------------------------------------------------------- host side
def preflight():
    ps.preflight(guards)
    rc, names, _ = ps.sh(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if rc != 0 or any(n.startswith("public-replay-") for n in names.splitlines()):
        raise guards.GuardError("a public-replay container exists or containers cannot be inspected")


def run_container(name, run, deadline):
    cname = f"public-replay-{name}-{uuid.uuid4().hex[:8]}"
    argv = ["docker", "run", "--rm", "--pull", "never", "--name", cname, "--gpus", "device=0",
            "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
            "-v", f"{ROOT}:{ROOT}:ro", "-v", f"{run}:{run}:rw", "--entrypoint", "python3.11", ps.IMAGE,
            str(ROOT / "public_replay.py"), "whisper-worker", "--config", name, "--out", str(run)]
    info, timeout = {"argv": argv, "container": cname}, min(WORKER_TIMEOUT_S, deadline - time.monotonic())
    with open(run / f"{name}.stdout.log", "w") as so, open(run / f"{name}.stderr.log", "w") as se:
        proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL)
        try:
            info["returncode"] = proc.wait(timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            info["error"] = f"worker timeout {timeout:.0f}s"
        finally:
            if proc.poll() is None:
                ps.sh(["docker", "kill", cname]), ps.sh(["docker", "rm", "-f", cname])
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill(), proc.wait(5)
            q = ["docker", "ps", "-a", "--filter", f"name=^/{cname}$", "--format", "{{.ID}}"]
            if ps.sh(q)[1]:
                ps.sh(["docker", "rm", "-f", cname])
            rc, left, _ = ps.sh(q)
            info["reaped"], info["container_removed"] = proc.returncode is not None, rc == 0 and not left
    if not info["container_removed"]:
        raise guards.GuardError(f"container {cname} could not be confirmed removed")
    if info.get("returncode") not in (0, None) and "error" not in info:
        info["error"] = f"worker exit {info['returncode']} (see {name}.stderr.log)"
    wj = run / f"{name}.worker.json"
    if wj.exists():
        info["worker"] = json.loads(wj.read_text())
        info["cold_init_ms"] = info["worker"]["cold_init_ms"]
    return info


def server_argv(prof, endpointing):
    ne, b = prof["native_endpointing"], lambda v: "true" if v else "false"
    return [prof["runtime"]["binary"], "serve", "--asr.model.path", prof["model"]["local_gguf_path"],
            "--device", "cuda:0", "--host", TEST_HOST, "--port", str(ps.PORT), "--no-ui",
            "--asr.batching.enabled=false", "--asr.streaming.rnnt_right_context", str(prof["rnnt_right_context"]),
            f"--asr.endpointing.enable={b(endpointing)}", f"--asr.endpointing.vad_based={b(ne['vad_based'])}",
            "--asr.endpointing.stop_history_eou_ms", str(ne["stop_history_eou_ms"])]


def check_server_log(text, rc):
    want = {"backend=CUDA0": r"backend=CUDA0\b", "mode=streaming": r"mode=streaming\b", f"right={rc}": rf"\bright={rc}\b"}
    want[f"step={(rc + 1) * 80}ms"] = rf"\bstep={(rc + 1) * 80}ms\b"
    return [k for k, pat in want.items() if not re.search(pat, text)]


def run_nemotron(name, run, deadline, cases):
    prof_name, mode, endpointing = CONFIGS[name]
    prof, pmeta = runner.load_profile(prof_name)
    prof, clock = copy.deepcopy(prof), audio.RealClock()
    prof["session"].update(verbatim=True, automatic_punctuation=True, word_timestamps=False)
    argv = server_argv(prof, endpointing)
    info = {"argv": argv, "profile_file": pmeta, "effective_profile": prof, "server_endpointing": endpointing,
            "rnnt_right_context": prof["rnnt_right_context"], "binary_sha256": ps.sha256(argv[0]),
            "gguf_sha256": ps.sha256(prof["model"]["local_gguf_path"]), "clock": clock.kind}
    logs = [run / f"{name}.server.stdout.log", run / f"{name}.server.stderr.log"]
    with open(logs[0], "w") as so, open(logs[1], "w") as se:
        t = time.perf_counter()
        proc = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            probe = adapters.LoopbackHTTP(prof["server"]["base_http"], 5)
            while True:  # readiness only; no offline HTTP warmup (the short replay is the warmup)
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited early rc={proc.returncode}")
                if time.perf_counter() - t > WORKER_TIMEOUT_S or time.monotonic() > deadline:
                    raise RuntimeError("server not ready within timeout/budget")
                try:
                    models = probe.get("/v1/models")
                    break
                except adapters.ProtocolError:
                    time.sleep(0.2)
            info["cold_init_ms"] = info["startup_to_ready_ms"] = (time.perf_counter() - t) * 1000
            info["models"] = models
            engine = adapters.NemotronEngine(prof, clock)
            engine.check_identity({"models": models})
            if not any(prof["server"]["expected_model_markers"][0] in m.get("id", "") and m.get("device") == "cuda:0"
                       for m in models.get("data", [])):
                raise RuntimeError("/v1/models does not report the model on exact device cuda:0")
            run_trials(name, engine, prof, mode, cases, clock, run, deadline)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"[:500]
        finally:  # owned process group only: SIGTERM, then SIGKILL, always reaped
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL), proc.wait(10)
            info["reaped"], info["exit_code"] = proc.returncode is not None, proc.returncode
    text = "".join(p.read_text(errors="replace") for p in logs)  # read after exit: buffered output is flushed
    info["server_log_effective"] = [l for l in text.splitlines() if l.startswith("[asr]")]
    bad = check_server_log(text, prof["rnnt_right_context"])
    if bad:
        info["error"] = (info.get("error") or "") + f" backend wrong/unverified: server log lacks {bad}"
    return info


def summarize(run, configs):
    out = {"anchor": ANCHOR, "note": "latency only, accuracy NOT assessed; nearest-rank percentiles over measured "
           "trials only; warmup and cold init excluded; no native throughput ranking is derived from offline runs; "
           "only the controlled table is an engine comparison", "controlled": {}, "native_800": {}}
    st = lambda v: {"n": len(v), "p50": ps.pctl(v, 50), "p95": ps.pctl(v, 95), "max": max(v) if v else None}
    for name, info in configs.items():
        p = run / f"{name}.trials.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        meas = [r for r in rows if r["phase"] == "measured"]
        ok = [r for r in meas if r["measured"]]
        lags = [r["max_send_lag_ms"] for r in rows if r.get("max_send_lag_ms") is not None]
        out["native_800" if CONFIGS[name][1] == "replay-native" else "controlled"][name] = {
            "n_ok": len(ok), "n_trials": len(meas), "expected": EXPECTED,
            "clip_end_to_final_ms": st([r["clip_end_to_final_ms"] for r in ok]),
            "per_sample": {c: st([r["clip_end_to_final_ms"] for r in ok if r["sample"] == c]) for c in CLIP_IDS},
            "status_counts": dict(collections.Counter(r["status"] for r in meas)),
            "multiple_auto_finals": sum(r["multiple_auto_finals"] for r in meas),
            "any_early_final": sum(r["any_early_final"] for r in meas),
            "anomalies": [r["anomaly"] for r in meas if r["anomaly"]],
            "max_send_lag_ms": max(lags) if lags else None,
            "errors": [{"round": r["round"], "sample": r["sample"], "error": r["error"]} for r in meas if r["error"]],
            "warmup": [{k: r[k] for k in ("status", "clip_end_to_final_ms", "error")} for r in rows
                       if r["phase"] == "warmup"],
            "cold_init_ms": info.get("cold_init_ms"), "vram": info.get("vram"), "error": info.get("error")}
    return out


def run_bench(a):
    if not a.allow_gpu:
        sys.exit("refusing: pass --allow-gpu (real GPU run)")
    start = time.monotonic()
    deadline = start + BUDGET_S
    cases = load_cases()
    run = ROOT / "results" / "public_replay" / f"{datetime.datetime.now(ps.JST):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    with guards.BenchLock() as lock:
        assert not os.get_inheritable(lock.f.fileno())  # docker/server children must not hold the lock
        preflight()
        rc, image_id, err = ps.sh(["docker", "image", "inspect", ps.IMAGE, "--format", "{{.Id}}"])
        if rc != 0:
            sys.exit(f"image {ps.IMAGE} not present ({err}); nothing is pulled or built")
        run.mkdir(parents=True, exist_ok=False)
        meta = {"started_at": ps.now(), "anchor": ANCHOR, "plan": plan(), "tail_ms": TAIL_MS, "packet_ms": 20,
                "budget_s": BUDGET_S, "image": ps.IMAGE, "image_id": image_id, "host_gpus": guards.query_gpus(),
                "cases": {k: {x: v for x, v in c.items() if x != "samples"} for k, c in cases.items()},
                "config_table": CONFIGS, "implementation_sha256": ps.sha256(__file__), "configs": {},
                "wsl_compute_process_accounting_limited": guards.is_wsl()}
        try:
            for name, (_, _, endpointing) in CONFIGS.items():
                if deadline - time.monotonic() < MIN_LEFT_S:
                    meta["configs"][name] = {"error": "skipped: run budget exhausted"}
                    continue
                preflight()
                baseline = ps.vram_used(guards)
                sampler = guards.VramSampler(0)
                sampler.start()
                try:
                    info = run_container(name, run, deadline) if endpointing is None else \
                        run_nemotron(name, run, deadline, cases)
                finally:
                    sampler.stop()
                info["vram"] = sampler.summary(baseline, "nvidia-smi before config start")
                meta["configs"][name] = info
                if not info.get("reaped", True):
                    raise guards.GuardError(f"{name}: process not reaped")
                ps.wait_released(guards, baseline, info)  # next model only after VRAM returned
        except guards.GuardError as e:
            meta["stopped"] = str(e)
            print(f"STOPPED: {e}", file=sys.stderr)
        finally:
            meta["finished_at"], meta["wall_s"] = ps.now(), round(time.monotonic() - start, 1)
            ps.write_json(run / "run.json", meta)
            summary = summarize(run, meta["configs"])
            ps.write_json(run / "summary.json", summary)
    for table in ("controlled", "native_800"):
        for name, s in summary[table].items():
            e = s["clip_end_to_final_ms"]
            print(f"{table:10s} {name:17s} n={s['n_ok']}/{EXPECTED} p50={e['p50']} p95={e['p95']} max={e['max']} "
                  f"lag_max={s['max_send_lag_ms']} status={s['status_counts']} err={len(s['errors'])} {s['error'] or ''}")
    print(f"results: {run}")
    failed = meta.get("stopped") or any(s["error"] or s["errors"] or s["n_ok"] != EXPECTED
                                        for t in ("controlled", "native_800") for s in summary[t].values())
    return 1 if failed or len(meta["configs"]) != len(CONFIGS) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("whisper-worker")
    w.add_argument("--config", choices=[k for k, v in CONFIGS.items() if v[2] is None], required=True)
    w.add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--allow-gpu", action="store_true")
    a = ap.parse_args(argv)
    return {"whisper-worker": whisper_worker, "run": run_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
