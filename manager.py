from typing import Tuple

Segment = Tuple[float | None, float | None, str]

class ASRManager:

    MIN_CHARS = 16

    def __init__(self):
        self.last_sent_text = ""

    def handle(self, seg: Segment, is_final: bool):

        beg, end, text = seg

        if beg is None or not text:
            return None

        text = text.strip()

        if not text:
            return None

        # partial 안정화 보호
        if len(text) < self.MIN_CHARS and not is_final:
            return None

        # 중복 제거
        if text == self.last_sent_text:
            return None

        delta = self._remove_prefix(text, self.last_sent_text)

        self.last_sent_text = text

        return delta

    def _remove_prefix(self, new: str, old: str):
        if old and new.startswith(old):
            return new[len(old):].strip()
        return new