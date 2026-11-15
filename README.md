# InterferenceEngine

[![CI](https://github.com/kapildhinwa2010-dev/InterferenceEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/InterferenceEngine/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/InterferenceEngine/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/InterferenceEngine/actions/workflows/codeql.yml)


Wave-lattice associative memory with interference recall and noisy-cue cleanup.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

WaveMemory stores bipolar (+/-1) patterns in a superposed wave
lattice and recalls them by interference: write_many() to load
patterns, recall() from a noisy cue, cleanup() with a flip-tolerance
check. Pure numpy.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from interferenceengine import WaveMemory

mem = WaveMemory(dim=64)
mem.write_many([p1, p2])
print(np.array_equal(mem.recall(noisy_cue), p1))
```

## API

- write / write_many / recall / cleanup
- hamming_distance(a, b)

## Honest results and limitations

- Interference recall degrades quickly as pattern count grows
  (cross-talk); capacity is a hint, not a measured law.
- The lattice dynamics are element-wise sign updates, not a full
  outer-product Hopfield network (see Memory-Lib for that).
- Negative result from the source experiments: carrier-mode recall
  under heavier noise failed about half the trials in the
  reserv.dean-part-2 log.

## License

MIT - see [LICENSE](LICENSE).
