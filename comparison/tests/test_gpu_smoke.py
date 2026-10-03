"""CPU-only mock tests for gpu_smoke guards/cleanup (no docker, no GPU, no nvidia-smi)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gpu_smoke  # noqa: E402
from asrbench import guards  # noqa: E402


class FakeRunner:
    def __init__(self, names="", vram=(1000, 1000)):
        self.names, self.vram, self.calls = names, list(vram), []

    def __call__(self, argv, timeout=15):
        self.calls.append(argv)
        if argv[:2] == ["docker", "ps"]:
            return 0, "" if "-aq" in argv else self.names, ""
        if argv[0] == "nvidia-smi":
            v = self.vram.pop(0) if len(self.vram) > 1 else self.vram[0]
            return 0, f"0, GPU-x, RTX, 1, 8192, {v}\n", ""
        return 0, "", ""


class FakePopen:
    def __init__(self, stdout="", hang=False, stderr="", interrupt=False):
        self.stdout_text, self.hang, self.killed, self.pid, self.returncode = stdout, hang, False, 999999, None
        self.stderr_text, self.interrupt = stderr, interrupt

    def __call__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        return self

    def communicate(self, timeout=None):
        if self.interrupt and not self.killed:
            raise KeyboardInterrupt
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("x", timeout)
        self.returncode = -9 if self.killed else 0
        return self.stdout_text, self.stderr_text

    def kill(self):
        self.killed = True


OK_LINE = gpu_smoke.MARK + json.dumps({"engine_device": "cuda", "transcript": "こんにちは"})
NEMO_OUT = json.dumps({"file": "x.wav", "text": "では今日のトークでは", "duration": 12})
NEMO_ERR = "[asr] model=nemotron-3.5-asr-streaming-0.6b.q8_0 head=rnnt backend=CUDA0\n"


class GpuSmokeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.proc = self.tmp.name
        self.args = SimpleNamespace(model="kotoba", allow_gpu=True, gpu=0, timeout=1, release_wait=0,
                                    lock=str(Path(self.tmp.name) / "lock"), cmd=[])
        self.exists = Path.exists
        orig = self.exists
        Path.exists = lambda p: True if str(p).startswith(str(gpu_smoke.ROOT / "local")) else orig(p)

    def tearDown(self):
        Path.exists = self.exists
        self.tmp.cleanup()

    def run_smoke(self, runner, popen):
        return gpu_smoke.smoke(self.args, runner=runner, popen=popen, sleep=lambda s: None, proc_root=self.proc)

    def test_requires_allow_gpu(self):
        self.args.allow_gpu = False
        with self.assertRaises(guards.GuardError):
            self.run_smoke(FakeRunner(), FakePopen())

    def test_refuses_when_live_service_or_other_smoke(self):
        for names in ("npc_whisper\n", "asrbench-smoke-abcd\n"):
            pop = FakePopen()
            with self.assertRaises(guards.GuardError):
                self.run_smoke(FakeRunner(names), pop)
            self.assertFalse(hasattr(pop, "argv"), "must not launch")

    def test_success_records_vram_and_cleanup(self):
        rep = self.run_smoke(FakeRunner(vram=(1000, 1050)), FakePopen(OK_LINE))
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["engine"]["engine_device"], "cuda")
        self.assertTrue(rep["vram"]["released"])
        self.assertNotIn(gpu_smoke.KOTOBA_SCRIPT, json.dumps(rep))

    def test_timeout_kills_only_own_container(self):
        r, pop = FakeRunner(), FakePopen(hang=True)
        rep = self.run_smoke(r, pop)
        self.assertFalse(rep["ok"])
        self.assertTrue(rep["timed_out"] and pop.killed)
        kills = [c for c in r.calls if c[:2] == ["docker", "kill"]]
        self.assertEqual(len(kills), 1)
        self.assertTrue(kills[0][2].startswith(gpu_smoke.SMOKE_PREFIX))

    def test_vram_not_released_fails(self):
        rep = self.run_smoke(FakeRunner(vram=(1000, 3000)), FakePopen(OK_LINE))
        self.assertFalse(rep["vram"]["released"])
        self.assertFalse(rep["ok"])

    def test_kotoba_uses_gpu0_only(self):
        pop = FakePopen(OK_LINE)
        self.run_smoke(FakeRunner(), pop)
        i = pop.argv.index("--gpus")
        self.assertEqual(pop.argv[i + 1], "device=0")
        self.assertEqual(pop.kw["cwd"], str(gpu_smoke.ROOT))

    def test_rejects_non_zero_gpu_before_launch(self):
        self.args.gpu = 1
        r, pop = FakeRunner(), FakePopen(OK_LINE)
        with self.assertRaises(guards.GuardError):
            self.run_smoke(r, pop)
        self.assertFalse(hasattr(pop, "argv"))
        self.assertEqual(r.calls, [], "must not query/monitor any GPU")

    def test_interrupt_cleans_own_container_and_reraises(self):
        r, pop = FakeRunner(), FakePopen(OK_LINE, interrupt=True)
        with self.assertRaises(KeyboardInterrupt):
            self.run_smoke(r, pop)
        self.assertTrue(pop.killed)
        kills = [c for c in r.calls if c[:2] == ["docker", "kill"]]
        self.assertEqual(len(kills), 1)
        self.assertTrue(kills[0][2].startswith(gpu_smoke.SMOKE_PREFIX))
        self.assertTrue(any(c[:3] == ["docker", "ps", "-aq"] for c in r.calls), "container check in finally")

    def test_nemotron_default_command_is_absolute_and_passes(self):
        self.args.model = "nemotron"
        pop = FakePopen(NEMO_OUT, stderr=NEMO_ERR)
        rep = self.run_smoke(FakeRunner(), pop)
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(pop.argv[0], str(gpu_smoke.abs_path(gpu_smoke.NEMO_BIN)))
        self.assertEqual(pop.argv[1], "transcribe")
        self.assertTrue(Path(pop.argv[2]).is_absolute())
        self.assertEqual(pop.argv[pop.argv.index("--device") + 1], "cuda:0")
        self.assertEqual(pop.argv[pop.argv.index("--format") + 1], "json")
        self.assertEqual(pop.argv[pop.argv.index("--model") + 1], str(gpu_smoke.NEMO_MODEL))
        self.assertEqual(rep["engine"]["transcript"], "では今日のトークでは")
        for k in ("cold_load_ms", "warmup_ms", "recognition_ms"):
            self.assertEqual(rep["engine"][k], "unknown")

    def test_nemotron_rejects_bad_argv_before_launch(self):
        self.args.model = "nemotron"
        base = gpu_smoke.nemotron_default_argv()

        def without(flag):
            i = base.index(flag)
            return base[:i] + base[i + 2:]
        bad = [
            ["/usr/bin/true"] + base[1:],                                  # binary outside runtime
            [base[0], "--help"],                                           # not transcribe
            [base[0]] + base[2:],                                          # missing subcommand
            without("--device"),                                           # implicit/auto device
            without("--device") + ["--device", "cpu"],
            without("--device") + ["--device", "cuda:1"],
            base + ["--device", "cuda:0"],                                 # duplicated
            base + ["--backend", "cpu"],                                   # alias override
            without("--format"),
            without("--format") + ["--format", "text"],
            without("--model"),                                            # default/downloaded model
            without("--model") + ["--model", "nemotron-3.5-asr-streaming-0.6b"],
            without("--model") + ["--model", "local/models/nemotron/model.bin"],
        ]
        for cmd in bad:
            pop = FakePopen(NEMO_OUT, stderr=NEMO_ERR)
            self.args.cmd = cmd
            with self.assertRaises(guards.GuardError, msg=cmd):
                self.run_smoke(FakeRunner(), pop)
            self.assertFalse(hasattr(pop, "argv"), cmd)

    def test_native_rejects_model_or_device_override_aliases(self):
        for extra in (["--asr.model.path", "remote-model"], ["--config", "extra.yaml"],
                      ["--diarize"], ["--asr.backend.gpu", "-1"]):
            with self.assertRaises(guards.GuardError):
                gpu_smoke.validate_native_argv(gpu_smoke.nemotron_default_argv() + extra)

    def test_nemotron_relative_paths_resolve_from_root(self):
        self.args.model = "nemotron"
        rel = [str(Path(a).relative_to(gpu_smoke.ROOT)) if a.startswith(str(gpu_smoke.ROOT)) else a
               for a in gpu_smoke.nemotron_default_argv()]
        self.args.cmd = rel
        pop = FakePopen(NEMO_OUT, stderr=NEMO_ERR)
        self.assertTrue(self.run_smoke(FakeRunner(), pop)["ok"])
        self.assertEqual(pop.argv[0], str(gpu_smoke.abs_path(gpu_smoke.NEMO_BIN)))
        self.assertEqual(pop.kw["cwd"], str(gpu_smoke.ROOT))

    def test_nemotron_output_must_be_json_text_on_cuda0(self):
        self.args.model = "nemotron"
        cases = {
            "help_stdout": FakePopen("Usage: nemo-speech transcribe <audio> [options]\n", stderr=NEMO_ERR),
            "empty_json": FakePopen("{}", stderr=NEMO_ERR),
            "empty_text": FakePopen(json.dumps({"text": "  "}), stderr=NEMO_ERR),
            "cpu_backend": FakePopen(NEMO_OUT, stderr=NEMO_ERR.replace("CUDA0", "CPU")),
            "no_backend_line": FakePopen(NEMO_OUT),
        }
        for name, pop in cases.items():
            rep = self.run_smoke(FakeRunner(), pop)
            self.assertFalse(rep["ok"], name)
            self.assertEqual(rep["returncode"], 0, name)


if __name__ == "__main__":
    unittest.main()
