# InterferenceEngine

InterferenceEngine is a small Python module for storing bipolar (`-1`/`+1`) vectors in an in-memory superposition and exploring retrieval and pattern overlap. Its current implementation is a basic associative-memory prototype, not a validated high-capacity or production retrieval system.

## Implemented features

`WaveMemory` provides these operations:

- `write()` and `write_many()` validate and add bipolar patterns to the memory's vector sum (the lattice), while retaining copies of each pattern.
- `recall()` selects the stored pattern with the largest dot product against a cue.
- `recall_with_scores()` returns the selected pattern, its overlap score, and the difference between the top two scores (or the sole score when there is one pattern).
- `recall_from_lattice()` makes a one-step sign update from the cue plus the superposed lattice, without consulting the stored copies. It is not guaranteed to return a stored pattern.
- `interference_matrix()` returns absolute, dimension-normalized pairwise overlaps for stored patterns.
- `cleanup()` calls `recall()` and raises `ValueError` if its result differs from the sign of the cue by more than the configured integer threshold (default: `8 * dim // 100` flips).
- `hamming_distance()` counts unequal entries in equally shaped arrays.

## Requirements and installation

The project metadata in `pyproject.toml` requires Python 3.9 or later and NumPy 1.24 or later. From a checkout, install the project with:

```bash
python -m pip install .
```

To install the test dependency as well, use:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from interferenceengine import WaveMemory

memory = WaveMemory(dim=8)
pattern = np.array([1, 1, 1, 1, -1, -1, -1, -1])
memory.write(pattern)

cue = pattern.copy()
cue[0] *= -1
recalled = memory.recall(cue)
print(np.array_equal(recalled, pattern))
print(memory.interference_matrix())
```

## Tests and CI

The test suite is run with:

```bash
python -m pytest -q
```

On the repository commit inspected for this draft (`ba4de25a4d106d7d33482d535bd7465fc76173aa`), this command completed with **17 passed**. The GitHub Actions CI workflow is configured to run tests and a compile/import smoke test on Python 3.10, 3.11, and 3.12; this local result does not establish the status of remote workflow runs.

## Limitations and status

- The memory is held in process memory; there is no persistence, file format, or data-loading API.
- `recall()` relies on a retained list of patterns. `recall_from_lattice()` instead applies a one-step sign operation to a superposition. Neither operation implements an iterative Hopfield-network update, and lattice cross-talk can prevent accurate recall as patterns interfere.
- The test suite checks selected small examples; the repository provides no benchmark suite or measured capacity/noise-robustness results. No performance or generalization guarantees are claimed.
- `noise_pattern(shape, flip_prob, rng)` validates `flip_prob` but currently returns an independently generated random bipolar array; it does not flip a fraction of an existing pattern. Do not treat it as a cue-corruption helper.
- The module has no command-line interface.

## License

The repository includes an MIT License in `LICENSE` (copyright 2026 Kapildhinwa2010-dev). See that file for the complete terms.
