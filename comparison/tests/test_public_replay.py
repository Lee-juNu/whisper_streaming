import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import public_replay as pr  # noqa: E402


def final(t, text, after):
    return {"t": t, "transcript": text, "empty": not text.strip(), "after_commit": after}


class PublicReplayTests(unittest.TestCase):
    def test_plan(self):
        steps = pr.plan()
        self.assertEqual(steps[0], ("warmup", 0, "short_a"))
        meas = [s for s in steps if s[0] == "measured"]
        self.assertEqual(len(meas), pr.EXPECTED)
        self.assertEqual(set(Counter(s[2] for s in meas).values()), {pr.REPEATS})
        self.assertEqual(meas[3][2], "medium_b")  # round 1 rotated by one

    def test_native_last_auto_final_before_commit(self):
        rec = {"finals": [final(103.0, "a", False), final(104.22, "b", False), final(106.0, "c", True),
                          final(106.1, "", True)]}
        t = pr.timing(rec, 100.0, 2500, "replay-native")
        self.assertAlmostEqual(t["clip_end_to_final_ms"], 1720.0, places=6)
        self.assertEqual(t["status"], "content_remaining_at_commit")
        self.assertTrue(t["multiple_auto_finals"])
        self.assertEqual(len(t["post_commit_final_ms"]), 1)
        self.assertAlmostEqual(rec["finals"][3]["t_rel_ms"], 6100.0, places=6)  # empty EOF final kept

    def test_empty_commit_does_not_invalidate_native_final(self):
        rec = {"finals": [final(104.0, "complete", False), final(106.0, "", True)]}
        self.assertEqual(pr.timing(rec, 100.0, 2500, "replay-native")["status"], "final")

    def test_native_no_auto_final_never_uses_eof_commit(self):
        rec = {"finals": [final(106.0, "text", True), final(106.1, "", True)]}
        t = pr.timing(rec, 100.0, 2500, "replay-native")
        self.assertIsNone(t["clip_end_to_final_ms"])
        self.assertEqual(t["status"], "no_native_final")

    def test_early_final_and_controlled_anomaly(self):
        rec = {"finals": [final(101.0, "a", False)]}
        self.assertEqual(pr.timing(rec, 100.0, 2500, "replay-native")["status"], "early_final")
        rec = {"finals": [final(101.0, "a", False), final(104.5, "ab", True)]}
        t = pr.timing(rec, 100.0, 2500, "replay-controlled")
        self.assertAlmostEqual(t["clip_end_to_final_ms"], 2000.0, places=6)
        self.assertTrue(t["any_early_final"])
        self.assertIsNotNone(t["anomaly"])

    def test_server_argv_and_log(self):
        prof = {"runtime": {"binary": "/b"}, "model": {"local_gguf_path": "/m.gguf"}, "rnnt_right_context": 3,
                "native_endpointing": {"vad_based": False, "stop_history_eou_ms": 800}}
        argv = pr.server_argv(prof, True)
        for a in ("--no-ui", "--asr.batching.enabled=false", "--asr.endpointing.enable=true",
                  "--asr.endpointing.vad_based=false"):
            self.assertIn(a, argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertEqual(argv[argv.index("--asr.streaming.rnnt_right_context") + 1], "3")
        self.assertIn("--asr.endpointing.enable=false", pr.server_argv(prof, False))
        log = "[asr] model=x head=rnnt backend=CUDA0\n[asr] mode=streaming head=rnnt left=56 center=1 right=3 step=320ms"
        self.assertEqual(pr.check_server_log(log, 3), [])
        self.assertEqual(pr.check_server_log(log, 1), ["right=1", "step=160ms"])
        self.assertIn("backend=CUDA0", pr.check_server_log(log.replace("CUDA0", "CPU"), 3))


if __name__ == "__main__":
    unittest.main()
