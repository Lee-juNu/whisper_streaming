import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import public_speed as ps  # noqa: E402


class PublicSpeedTests(unittest.TestCase):
    def test_plan_rotated_balanced(self):
        ids = [c[0] for c in ps.CLIPS]
        steps = ps.plan(ids, 8)
        self.assertEqual(steps[:2], [("warmup", 0, "long_b"), ("warmup", 1, "long_b")])
        warm = [s for s in steps if s[0] == "warm"]
        self.assertEqual(len(warm), 48)
        self.assertEqual(set(Counter(s[2] for s in warm).values()), {8})
        self.assertEqual(warm[6][2], "short_b")  # round 1 starts one clip later
        self.assertEqual(steps, ps.plan(ids, 8))

    def test_nearest_rank(self):
        self.assertEqual(ps.pctl([5, 1, 3, 2, 4], 50), 3)
        self.assertEqual(ps.pctl(list(range(1, 21)), 95), 19)
        self.assertEqual(ps.pctl([7], 95), 7)
        self.assertIsNone(ps.pctl([], 50))

    def test_multipart(self):
        body, ctype = ps.multipart({"language": "ja-JP"}, "a.wav", b"RIFFxx")
        b = ctype.split("boundary=")[1]
        self.assertIn(b'name="language"\r\n\r\nja-JP', body)
        self.assertIn(b'filename="a.wav"', body)
        self.assertTrue(body.endswith(f"--{b}--\r\n".encode()))


if __name__ == "__main__":
    unittest.main()
