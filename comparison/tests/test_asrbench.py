"""stdlib-only tests: python -m unittest discover -s tests -v  (run from the comparison/ dir)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asrbench import ROOT, __main__ as cli, audio, dataset, guards, metrics, mock, postfilter, runner  # noqa: E402
from asrbench.adapters import RecognitionInput  # noqa: E402

TMP = DS = ITEMS = None


def setUpModule():
    global TMP, DS, ITEMS
    TMP = tempfile.TemporaryDirectory()
    DS = mock.write_synthetic_dataset(Path(TMP.name) / "data")
    ITEMS, issues = dataset.load_dataset(DS)
    assert not dataset.has_errors(issues), issues


def tearDownModule():
    TMP.cleanup()


def mock_run(profile_name, mode, items=None):
    prof, meta = runner.load_profile(profile_name)
    clock = audio.FakeClock()
    eng = (mock.MockWhisperEngine(clock) if prof["engine"] == "faster_whisper"
           else mock.MockNemotronEngine(clock, server_endpointing=mode == "replay-native"))
    out, summary = runner.run_benchmark(prof, meta, mode, items or ITEMS, eng, clock, Path(TMP.name) / "runs",
                                        dataset_path=DS)
    with open(out / "utterances.jsonl", encoding="utf-8") as f:
        recs = {r["utt_id"]: r for r in map(json.loads, f)}
    return out, summary, recs, eng


class MetricsTest(unittest.TestCase):
    def test_cer_and_missing_reference(self):
        self.assertEqual(metrics.utterance_cer("みーちゃん、おはよう。", "みーちゃんおはよ")["edits"], 1)
        self.assertIsNone(metrics.utterance_cer(None, "x"))
        a = metrics.cer_aggregate([("ab", "ab"), (None, "x"), ("abcd", None)])
        self.assertEqual((a["micro_cer"], a["no_reference"], a["failed_with_reference"]), (0.0, 1, 1))

    def test_nickname_longest_match(self):
        r = metrics.nickname_utterance("タロウとタロ", ["タロウ", "タロ"], {"タロウ": 1})
        self.assertEqual((r["hits"], r["false_insertions"]), (1, 1))


class EndpointTest(unittest.TestCase):
    cfg = audio.EndpointConfig()

    def test_leading_quiet_audio_starts_buffer(self):
        s = audio.tone(300, amp=0.005) + audio.tone(500) + audio.silence(2000)
        segs, short = audio.find_segments(s, self.cfg)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].start, 0)  # quiet (>= .001, < .02) lead-in is buffered
        self.assertEqual(segs[0].voice_end, audio.ms_to_samples(800))

    def test_all_zero_never_opens_buffer(self):
        self.assertEqual(audio.find_segments(audio.silence(3000), self.cfg), ([], []))

    def test_min_segment_is_buffer_length(self):
        segs, short = audio.find_segments(audio.tone(100) + audio.silence(2000), self.cfg)
        self.assertEqual((len(segs), len(short)), (1, 0))  # 100 ms voice, 1600 ms buffer -> kept

    def test_short_buffer_filtered_without_grading_impact(self):
        segs, short = audio.find_segments(audio.silence(1000) + audio.tone(200), self.cfg)
        self.assertEqual((len(segs), len(short)), (0, 1))
        self.assertEqual(audio.classify_single(segs), (None, "no_segment_detected"))

    def test_quiet_only_buffer_has_no_voice(self):
        segs, _ = audio.find_segments(audio.tone(2000, amp=0.005), self.cfg)
        self.assertEqual(segs[0].voice_end, segs[0].start)


class RunnerTest(unittest.TestCase):
    def test_file_controlled_has_no_latency(self):
        _, s, recs, _ = mock_run("whisper-small", "file-controlled")
        for r in recs.values():
            self.assertIsNone(r["final_latency_ms"])
            self.assertIsNone(r["first_partial_latency_ms"])
            self.assertEqual(r["processing_ms"], 150.0)
        self.assertEqual(s["final_latency_ms"]["n"], 0)

    def test_replay_controlled_whisper(self):
        out, s, recs, eng = mock_run("whisper-small", "replay-controlled")
        self.assertAlmostEqual(recs["syn-s1-001"]["final_latency_ms"], 3200 - 1700 + 150, places=3)
        self.assertEqual(recs["syn-s1-003"]["hypothesis"], "")
        self.assertIsNone(recs["syn-s1-003"]["failure"])
        self.assertEqual(recs["syn-s2-001"]["failure"], "endpoint:multiple_segments")
        self.assertEqual(recs["syn-s2-002"]["failure"], "endpoint:exceeds_max_segment")
        self.assertEqual(recs["syn-s2-003"]["failure"], "endpoint:insufficient_trailing_silence")
        self.assertEqual(len(s["failures"]), 3)
        self.assertEqual(recs["syn-s1-002"]["prompt"], "みーちゃんおはよ")  # previous hypothesis, not truth
        self.assertEqual(s["cer"]["no_reference"], 1)
        nk = s["nickname"]
        self.assertAlmostEqual(nk["mention_recall"], 2 / 3)
        self.assertAlmostEqual(nk["candidate_list_precision"], 2 / 3)
        meta = json.loads((out / "run.json").read_text(encoding="utf-8"))["meta"]
        self.assertTrue(meta["mock"])
        self.assertIn("MOCK", meta["label"])
        self.assertEqual(meta["lock"]["model"]["revision"], "536b0662742c02347bc0e980a01041f333bce120")

    def test_truth_never_reaches_engine(self):
        names = {f.name for f in fields(RecognitionInput)}
        self.assertFalse(names & {"reference", "spoken_nicknames", "speech_start_ms", "speech_end_ms", "truth"})
        items = [replace(it, truth=replace(it.truth, reference=f"TRUTH-SENTINEL-{i}",
                                           spoken_nicknames={"SENTINEL-NAME": 1}))
                 for i, it in enumerate(ITEMS)]
        for mode in ("file-controlled", "replay-controlled"):
            _, _, recs, eng = mock_run("whisper-small", mode, items)
            dumped = json.dumps(eng.calls, ensure_ascii=False)
            self.assertNotIn("SENTINEL", dumped)
            self.assertEqual(recs["syn-s1-001"]["truth"]["reference"], "TRUTH-SENTINEL-0")  # scored afterwards

    def test_replay_native_multiple_finals_and_empty_flush(self):
        out, s, recs, _ = mock_run("nemotron-rc1-160ms", "replay-native")
        r = recs["syn-s2-001"]
        self.assertEqual(r["hypothesis"], "ふたつのはつわポチ")
        self.assertEqual((r["n_finals"], r["n_content_finals"]), (3, 2))
        self.assertTrue(r["finals"][-1]["empty"] and r["finals"][-1]["after_commit"])
        self.assertAlmostEqual(r["final_latency_ms"], 920, places=3)  # empty flush did not replace it
        self.assertTrue(r["finals"][0]["early"] and r["any_early_final"])
        self.assertFalse(r["early_final"])
        self.assertTrue(r["multiple_finals"])
        self.assertAlmostEqual(r["first_partial_latency_ms"], 360, places=3)
        self.assertIn("syn-s2-001", s["multiple_final_utterances"])
        events = (out / "events.jsonl").read_text(encoding="utf-8")
        self.assertIn("input_audio_buffer.committed", events)

    def test_replay_controlled_nemotron(self):
        _, _, recs, _ = mock_run("nemotron-rc1-160ms", "replay-controlled")
        self.assertAlmostEqual(recs["syn-s1-001"]["final_latency_ms"], 3200 - 1700 + 120, places=3)

    def test_rescore_and_refuse_overwrite(self):
        out, s, _, _ = mock_run("whisper-small", "replay-controlled")
        path, s2 = runner.rescore(out)
        self.assertEqual(s2["cer"], s["cer"])
        path2, _ = runner.rescore(out, ITEMS)
        self.assertNotEqual(path, path2)
        with self.assertRaises(FileExistsError):
            runner.make_run_dir(out.parent, out.name)


class GuardsTest(unittest.TestCase):
    def test_scan_stores_process_name_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "4242")
            p.mkdir()
            (p / "cmdline").write_bytes(b"python3\0ws_server.py\0--api-key\0SECRET123\0")
            (p / "comm").write_text("python3\n")
            hits = guards.scan_cmdlines(set(), proc_root=d)
        self.assertEqual(hits, [{"pid": 4242, "name": "python3", "matched": ["ws_server"]}])
        res = guards.assess_interference({"ok": True, "apps": [], "pid_unknown": False}, {1}, False, False, hits)
        self.assertTrue(res["blocked"])
        self.assertNotIn("SECRET123", json.dumps(res))

    def test_server_command_print_only(self):
        prof, _ = runner.load_profile("nemotron-rc1-160ms")
        cmd = guards.server_command(prof, "replay-native", binary="/opt/nemo-speech")
        self.assertEqual(cmd[1:6], ["serve", "--host", "127.0.0.1", "--port", "18101"])
        self.assertIn("--asr.endpointing.enable=true", cmd)
        self.assertEqual(cmd[cmd.index("--asr.endpointing.stop_history_eou_ms") + 1], "800")
        ctl = guards.server_command(prof, "replay-controlled", binary="/opt/nemo-speech")
        self.assertIn("--asr.endpointing.enable=false", ctl)

    def test_postfilter_gap_is_explicit(self):
        prof, _ = runner.load_profile("kotoba-whisper-v2")
        meta = postfilter.load_postfilter(prof["postfilter"], ROOT).meta
        self.assertEqual(meta["parity"], "UNVERIFIED_GAP")
        self.assertIn("NOT applied", meta["profile_note"])


class CliTest(unittest.TestCase):
    def call(self, *argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = cli.main(list(argv))
        return rc, buf.getvalue(), err.getvalue()

    def test_run_requires_allow_inference(self):
        rc, _, err = self.call("run", "--profile", "whisper-small", "--mode", "file-controlled",
                               "--dataset", str(DS))
        self.assertEqual(rc, 2)
        self.assertIn("--allow-inference", err)

    def test_preflight_rejects_placeholder_dataset(self):
        rc, out, _ = self.call("preflight", "--dataset", str(ROOT / "examples" / "dataset.example.jsonl"))
        self.assertEqual(rc, 1)
        self.assertIn("placeholder", out)
        rc, _, _ = self.call("preflight", "--dataset", str(DS))
        self.assertEqual(rc, 0)

    def test_server_command_cli(self):
        rc, out, _ = self.call("server-command", "--profile", "nemotron-rc3-320ms", "--mode", "replay-controlled")
        self.assertEqual(rc, 0)
        self.assertIn("--asr.streaming.rnnt_right_context 3", out)
        self.assertEqual(self.call("server-command", "--profile", "whisper-small",
                                   "--mode", "file-controlled")[0], 2)

    def test_mock_cli_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.call("mock", "--out-dir", d)
            self.assertEqual(rc, 0)
            res = json.loads(out)
            self.assertEqual(len(res["runs"]), 5)
            run_dir = res["runs"]["nemotron-rc1-160ms/replay-native"]["dir"]
            self.assertEqual(self.call("score", run_dir)[0], 0)
        self.assertNotIn("ws_server", sys.modules)


if __name__ == "__main__":
    unittest.main()
