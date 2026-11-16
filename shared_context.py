from __future__ import annotations
import numpy as np


class SharedContextPredictor:
    def __init__(self, vocab_size: int, num_streams: int, max_order: int = 3,
                 alpha: float = 0.5, strengths=(12.0, 8.0, 4.0, 2.0, 1.0)):
        if not 1 <= vocab_size <= 256:
            raise ValueError("vocab_size should be between 1 and 256")
        if not 1 <= max_order <= len(strengths):
            raise ValueError("max_order must be <= len(strengths) and positive")
        if num_streams < 1 or alpha <= 0 or any(s <= 0 for s in strengths):
            raise ValueError("invalid predictor parameters")
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.alpha = float(alpha)
        self.strengths = tuple(float(s) for s in strengths)
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self.total = 0
        self.histories = [bytearray() for _ in range(num_streams)]
        # Entries map context bytes to [occurrence_count, {next_symbol:count}]
        self.tables = [None] + [dict() for _ in range(max_order)]

    def predict(self, stream: int) -> np.ndarray:
        p = (self.counts.astype(np.float64) + self.alpha)
        p /= self.total + self.alpha * self.vocab_size
        history = self.histories[stream]
        for order in range(1, min(len(history), self.max_order) + 1):
            entry = self.tables[order].get(bytes(history[-order:]))
            if entry is None:
                continue
            n, next_counts = entry
            s = self.strengths[order-1]
            p *= s / (n + s)
            for symbol, count in next_counts.items():
                p[symbol] += count / (n + s)
        return p

    def update(self, stream: int, symbol: int) -> None:
        if not 0 <= symbol < self.vocab_size:
            raise ValueError("symbol index out of vocabulary")
        history = self.histories[stream]
        for order in range(1, min(len(history), self.max_order) + 1):
            key = bytes(history[-order:])
            table = self.tables[order]
            if key not in table:
                table[key] = [0, {}]
            item = table[key]
            item[0] += 1
            freq = item[1]
            freq[symbol] = freq.get(symbol, 0) + 1
        self.counts[symbol] += 1
        self.total += 1
        history.append(symbol)
        if len(history) > self.max_order:
            del history[0]
