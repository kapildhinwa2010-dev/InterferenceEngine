# InterferenceEngine - Interference analysis and lattice recall over wave memory
# Part of the Kapildhinwa2010-dev Toolkit.

"""Interference analysis for superposition (wave-lattice) memory.

Lineage: reserv.dean-part-2 (helix.py wave arena). Patterns are stored
both as copies and as a superposed lattice (sum of bipolar patterns).
This repo is the ANALYSIS layer: it quantifies cross-talk between stored
patterns (interference_matrix), reports recall confidence (recall_with_scores),
and recalls from the lattice alone (recall_from_lattice).

HONEST LIMITATIONS: one-step lattice recall is exact only while stored
patterns stay near-orthogonal; cross-talk grows with the number of
superposed patterns and recall accuracy degrades well before the
theoretical capacity. Our experiments measured poor noise robustness for
carrier-style lattice recall - see the README for the numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WaveMemory",
    "hamming_distance",
    "noise_pattern",
]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("patterns must have the same shape")
    return int((a != b).sum())


def noise_pattern(shape, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Random +/-1 pattern."""
    if not 0.0 <= flip_prob <= 1.0:
        raise ValueError("flip_prob must be in [0, 1]")
    return rng.choice(np.array([-1, 1]), size=shape)


class WaveMemory:
    """Superposition memory with interference analysis and lattice recall."""

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.lattice = np.zeros(self.dim, dtype=np.float64)
        self.patterns: list = []

    def write(self, pattern: np.ndarray) -> None:
        pattern = np.asarray(pattern, dtype=np.float64).ravel()
        if pattern.size != self.dim:
            raise ValueError("pattern size must equal dim")
        if not np.all(np.isin(pattern, (-1.0, 1.0))):
            raise ValueError("patterns must contain only -1 and +1")
        self.lattice += pattern
        self.patterns.append(pattern.copy())

    def write_many(self, patterns) -> None:
        for pattern in patterns:
            self.write(pattern)

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """Return the stored pattern most correlated with ``cue``."""
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns:
            raise ValueError("memory is empty")
        scores = [float(cue @ p) for p in self.patterns]
        return self.patterns[int(np.argmax(scores))].copy()

    def recall_from_lattice(self, cue: np.ndarray) -> np.ndarray:
        """One-step recall from the superposed lattice ALONE (no copies).

        Computes sign(cue + lattice). Exact when stored patterns are
        near-orthogonal and few; degrades with cross-talk as the pattern
        count grows. This is a real limitation of superposition, not a bug.
        """
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not np.any(self.lattice):
            raise ValueError("memory is empty")
        state = np.sign(cue + self.lattice)
        state[state == 0] = 1.0
        return state

    def recall_with_scores(self, cue: np.ndarray) -> dict:
        """Best pattern plus its overlap score and margin over the runner-up.

        A small margin means the cue is ambiguous (interference); callers
        should treat low-margin recalls as unreliable.
        """
        cue = np.asarray(cue, dtype=np.float64).ravel()
        if cue.size != self.dim:
            raise ValueError("cue size must equal dim")
        if not self.patterns:
            raise ValueError("memory is empty")
        scores = sorted((float(cue @ p) for p in self.patterns), reverse=True)
        best = self.recall(cue)
        margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
        return {"pattern": best, "score": scores[0], "margin": margin}

    def interference_matrix(self) -> np.ndarray:
        """Normalized pairwise overlap |pi . pj| / dim between stored patterns.

        Diagonal is 1; off-diagonal entries near 0 mean low interference.
        """
        if not self.patterns:
            raise ValueError("memory is empty")
        stacked = np.stack(self.patterns)
        overlaps = stacked @ stacked.T / self.dim
        return np.abs(overlaps)

    def cleanup(self, noisy: np.ndarray, max_flips: int = 8) -> np.ndarray:
        """Recall and verify; raise if the result is too far from the cue."""
        recovered = self.recall(noisy)
        flips = hamming_distance(recovered, np.sign(noisy))
        if flips > max_flips * self.dim // 100:
            raise ValueError("cleanup failed: {} flips".format(flips))
        return recovered

    def __len__(self) -> int:
        return len(self.patterns)