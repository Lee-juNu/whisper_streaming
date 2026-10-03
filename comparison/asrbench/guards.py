"""Safety guards: placeholders, loopback-only endpoint, sequential lock, GPU interference
checks, nvidia-smi VRAM sampling and native-server attestation. Nothing here starts or
stops services; nvidia-smi is only called when a caller explicitly asks for it."""
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

from . import TEST_HOST, TEST_PORT

PLACEHOLDER_RE = re.compile(r"<[A-Za-z_][^<>]*>")
DEFAULT_LOCK = Path(tempfile.gettempdir()) / "asrbench-gpu.lock"
LIVE_SERVICE_PATTERNS = ("ws_server", "nemo-speech", "nemo_speech")
GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
             "--format=csv,noheader,nounits"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
              "--format=csv,noheader,nounits"]
LOOPBACK_HEX = {"0100007F", "00000000000000000000000001000000", "0000000000000000FFFF00000100007F"}
VRAM_LABEL = ("sampled TOTAL device memory.used (MiB) of the whole GPU via nvidia-smi (all processes, "
              "not a per-model allocation); peaks shorter than the sampling interval can be missed")


class GuardError(RuntimeError):
    """Blocking safety condition; the message says what the operator must do."""


def find_placeholders(obj):
    if isinstance(obj, str):
        return PLACEHOLDER_RE.findall(obj)
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in find_placeholders(v)]
    if isinstance(obj, (list, tuple)):
        return [p for v in obj for p in find_placeholders(v)]
    return []


def validate_loopback_url(url, schemes):
    u = urllib.parse.urlsplit(url)
    try:
        port = u.port
    except ValueError:
        port = None
    if u.scheme not in schemes or u.hostname != TEST_HOST or port != TEST_PORT:
        raise GuardError(f"{url}: only {'/'.join(schemes)}://{TEST_HOST}:{TEST_PORT} (dedicated loopback "
                         "test endpoint) is allowed; never point the harness at a live service")
    return url


def is_wsl(proc_version="/proc/version"):
    try:
        return "microsoft" in Path(proc_version).read_text().lower()
    except OSError:
        return False


def run_cmd(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, "", f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return None, "", f"{argv[0]}: timed out after {timeout}s"
    return p.returncode, p.stdout, p.stderr


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def query_gpus(runner=run_cmd):
    rc, out, err = runner(GPU_QUERY)
    if rc != 0:
        raise GuardError(f"nvidia-smi GPU query failed ({(err or '').strip() or rc}). Real runs need "
                         "nvidia-smi for GPU identity and VRAM sampling; nothing was started.")
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6 and p[0].isdigit():
            gpus.append({"index": int(p[0]), "uuid": p[1], "name": p[2], "driver": p[3],
                         "memory_total_mib": _num(p[4]), "memory_used_mib": _num(p[5])})
    if not gpus:
        raise GuardError(f"nvidia-smi returned no GPUs: {out!r}")
    return gpus


def query_compute_apps(runner=run_cmd):
    rc, out, err = runner(APPS_QUERY)
    if rc != 0:
        return {"ok": False, "apps": [], "pid_unknown": True, "error": (err or "").strip() or str(rc)}
    apps, unknown = [], False
    for line in out.splitlines():
        s = line.strip()
        if not s or s.lower().startswith("no running"):
            continue
        p = [x.strip() for x in s.split(",")]
        pid = int(p[0]) if p[0].isdigit() else None
        unknown |= pid is None
        apps.append({"pid": pid, "name": p[1] if len(p) > 1 else None,
                     "used_mib": _num(p[2]) if len(p) > 2 else None, "gpu_uuid": p[3] if len(p) > 3 else None})
    return {"ok": True, "apps": apps, "pid_unknown": unknown, "error": None}


def ancestor_pids(pid, proc_root="/proc"):
    out = set()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            status = Path(proc_root, str(pid), "status").read_text()
        except OSError:
            break
        m = re.search(r"^PPid:\s*(\d+)", status, re.M)
        pid = int(m.group(1)) if m else 0
    return out


def process_name(pid, proc_root="/proc"):
    """Kernel process name only (/proc/<pid>/comm). Full command lines are never stored,
    since they may carry credentials/tokens in arguments."""
    try:
        return Path(proc_root, str(pid), "comm").read_text(errors="replace").strip()[:64] or None
    except OSError:
        return None


def scan_cmdlines(exclude, patterns=LIVE_SERVICE_PATTERNS, proc_root="/proc"):
    """Match patterns against cmdlines in memory; record only pid, process name and pattern."""
    hits = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return hits
    for d in entries:
        if not d.isdigit() or int(d) in exclude:
            continue
        try:
            cmd = Path(proc_root, d, "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        matched = [p for p in patterns if p in cmd]
        if matched:
            hits.append({"pid": int(d), "name": process_name(d, proc_root), "matched": matched})
    return hits


def assess_interference(apps_res, allowed_pids, wsl, attest_exclusive, cmdline_hits, required_pids=()):
    blocks, notes = [], []
    seen = {a["pid"] for a in apps_res["apps"]}
    others = [a for a in apps_res["apps"] if a["pid"] not in allowed_pids]
    if others:
        blocks.append("other GPU compute processes are running: "
                      + ", ".join(f"pid={a['pid']} {a['name']}" for a in others)
                      + ". Stop them yourself (this harness never stops services) or wait for an exclusive GPU window.")
    for h in cmdline_hits:
        blocks.append(f"pid {h['pid']} looks like a live ASR service (process {h['name']!r}, "
                      f"matched {h['matched']}). "
                      "Stop it manually first, or launch the test server directly (not via a wrapper shell).")
    verifiable = apps_res["ok"] and not wsl and not apps_res["pid_unknown"]
    if verifiable:
        missing = [p for p in required_pids if p not in seen]
        if missing:
            blocks.append(f"permitted server PID(s) {missing} not visible as GPU compute process: wrong "
                          "--server-pid, or the server is not using the GPU")
    else:
        reason = ("WSL2: nvidia-smi generally cannot attribute GPU memory to processes, so an empty list "
                  "does not prove the GPU is idle" if wsl else
                  f"GPU process accounting unavailable ({apps_res.get('error') or 'unparseable PIDs'})")
        if attest_exclusive:
            notes.append(reason + "; operator attested exclusivity (--operator-attest-gpu-exclusive)")
        else:
            blocks.append(reason + ". Failing closed. After confirming manually (e.g. Windows Task Manager / "
                          "host nvidia-smi) that only the permitted process uses the GPU, rerun with "
                          "--operator-attest-gpu-exclusive; the attestation is recorded in every output.")
    status = ("blocked" if blocks else "verified_exclusive" if verifiable else "operator_attested_unverifiable")
    return {"status": status, "blocked": bool(blocks), "blocks": blocks, "notes": notes,
            "allowed_pids": sorted(allowed_pids), "wsl": wsl, "compute_apps": apps_res,
            "cmdline_hits": cmdline_hits}


def listeners(port, proc_root="/proc"):
    res = []
    for fam in ("tcp", "tcp6"):
        try:
            rows = Path(proc_root, "net", fam).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) < 10 or ":" not in f[1]:
                continue
            ip, port_hex = f[1].rsplit(":", 1)
            if int(port_hex, 16) == port and f[3] == "0A":
                res.append({"family": fam, "ip_hex": ip.upper(), "loopback": ip.upper() in LOOPBACK_HEX,
                            "inode": f[9]})
    return res


def verify_server_pid(pid, port=TEST_PORT, proc_root="/proc"):
    ls = listeners(port, proc_root)
    if not ls:
        raise GuardError(f"nothing is listening on {TEST_HOST}:{port}. Start the dedicated test server "
                         "manually with the command printed by `server-command`.")
    bad = [l for l in ls if not l["loopback"]]
    if bad:
        raise GuardError(f"port {port} is bound to a non-loopback address {bad}; restart the test server "
                         f"bound to {TEST_HOST} only")
    fd_dir = Path(proc_root, str(pid), "fd")
    try:
        names = os.listdir(fd_dir)
    except FileNotFoundError:
        raise GuardError(f"--server-pid {pid} does not exist")
    except PermissionError:
        raise GuardError(f"cannot inspect {fd_dir} (permission). Run the test server as the same user; "
                         "failing closed instead of trusting the PID")
    owned = set()
    for n in names:
        try:
            m = re.fullmatch(r"socket:\[(\d+)\]", os.readlink(fd_dir / n))
        except OSError:
            continue
        if m:
            owned.add(m.group(1))
    if not owned & {l["inode"] for l in ls}:
        raise GuardError(f"PID {pid} does not own the listener on {TEST_HOST}:{port}; pass the PID of the "
                         "test server process itself")
    return {"pid": pid, "listeners": ls, "name": process_name(pid, proc_root)}


def nemotron_server_args(profile, mode):
    """Exact native server settings a mode requires (dotted engine args from pinned docs)."""
    args = ["--asr.model.path", profile["model"]["local_gguf_path"],
            "--asr.backend.gpu", str(profile.get("gpu_index", 0)),
            "--asr.streaming.rnnt_right_context", str(profile["rnnt_right_context"]),
            "--asr.batching.enabled=false"]
    if mode == "replay-native":
        ne = profile["native_endpointing"]
        args += ["--asr.endpointing.enable=true",
                 f"--asr.endpointing.vad_based={'true' if ne['vad_based'] else 'false'}",
                 "--asr.endpointing.stop_history_eou_ms", str(ne["stop_history_eou_ms"])]
    else:
        args += ["--asr.endpointing.enable=false"]
    return args


def server_command(profile, mode, binary=None):
    binary = binary or profile["runtime"]["binary"]
    # --host/--port aliases verified in docs/server.md at the pinned commit (lock.json "server").
    return [binary, "serve", "--host", TEST_HOST, "--port", str(TEST_PORT)] + nemotron_server_args(profile, mode)


ATTEST_REQUIRED = ("nemo_speech_commit", "gguf_path", "serve_command", "rnnt_right_context",
                   "endpointing_enable", "batching_enabled", "speech_contexts_supported",
                   "attested_by", "attested_at")


def check_attestation(att, profile, mode, bias, lock):
    """Server endpointing/context settings cannot be read back over the API, so the operator
    attests them; we cross-check the attestation against the profile and pinned lock."""
    errs = [f"attestation missing {k!r}" for k in ATTEST_REQUIRED if k not in att]
    if errs:
        return errs
    commit = lock["sources"]["nemo_speech_cpp"]["commit"]
    if att["nemo_speech_commit"] != commit:
        errs.append(f"server built from {att['nemo_speech_commit']!r}, lock pins {commit}")
    g = Path(att["gguf_path"])
    if g.name != profile["model"]["gguf_filename"] or "speech-streaming-en" in str(g):
        errs.append(f"GGUF {g.name!r} is not {profile['model']['gguf_filename']} (English model is rejected)")
    if not g.is_file():
        errs.append(f"attested GGUF not found: {g}")
    elif Path(profile["model"]["local_gguf_path"]).resolve() != g.resolve():
        errs.append("attested GGUF differs from profile model.local_gguf_path")
    if att["rnnt_right_context"] != profile["rnnt_right_context"]:
        errs.append(f"rnnt_right_context {att['rnnt_right_context']} != profile {profile['rnnt_right_context']}")
    if att["batching_enabled"] is not False:
        errs.append("batching must be disabled (single stream)")
    native = mode == "replay-native"
    if att["endpointing_enable"] is not native:
        errs.append(f"mode {mode} needs server endpointing {'enabled' if native else 'DISABLED'}")
    if native:
        ne = profile["native_endpointing"]
        if att.get("vad_based") is not ne["vad_based"] or att.get("stop_history_eou_ms") != ne["stop_history_eou_ms"]:
            errs.append(f"native endpointing must be vad_based={ne['vad_based']} "
                        f"stop_history_eou_ms={ne['stop_history_eou_ms']}")
    if bias == "on" and att["speech_contexts_supported"] is not True:
        errs.append("bias on needs speech_contexts_supported=true (operator-verified tokenizer-supported GGUF)")
    cmd = att["serve_command"]
    missing = [a for a in nemotron_server_args(profile, mode) if a not in cmd]
    if missing:
        errs.append(f"attested serve_command lacks {missing}")
    return errs


class BenchLock:
    """Non-blocking exclusive flock so only one benchmark (one model) runs at a time."""

    def __init__(self, path=DEFAULT_LOCK):
        self.path = Path(path)
        self.f = None

    def __enter__(self):
        import fcntl
        self.f = open(self.path, "a+")
        try:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.f.close()
            self.f = None
            raise GuardError(f"another benchmark holds {self.path}; run profiles sequentially (one model at a time)")
        self.f.seek(0)
        self.f.truncate()
        self.f.write(f"{os.getpid()}\n")
        self.f.flush()
        return self

    def __exit__(self, *exc):
        import fcntl
        if self.f:
            fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            self.f.close()
            self.f = None


class VramSampler:
    def __init__(self, gpu_index, interval_ms=100, popen=subprocess.Popen, clock=time.monotonic):
        self.gpu_index, self.interval_ms, self.popen, self.clock = gpu_index, interval_ms, popen, clock
        self.samples, self.errors = [], []
        self.proc = self.thread = None

    def start(self):
        self.proc = self.popen(["nvidia-smi", "-i", str(self.gpu_index), "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            v = _num(line.strip())
            if v is None:
                self.errors.append(line.strip()[:200])
            else:
                self.samples.append((self.clock(), v))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.thread.join(5)

    def summary(self, baseline_mib, baseline_source):
        vals = [v for _, v in self.samples]
        mx = max(vals) if vals else None
        return {"label": VRAM_LABEL, "interval_ms": self.interval_ms, "n_samples": len(vals),
                "baseline_mib": baseline_mib, "baseline_source": baseline_source, "max_mib": mx,
                "max_minus_baseline_mib": (mx - baseline_mib) if mx is not None and baseline_mib is not None else None,
                "sampler_errors": self.errors[:20]}
