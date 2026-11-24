# t=21 — Grounded Passage Citation Verifier & Abstention DPO Pair Synthesizer

Standalone Python 3.12, zero API calls, zero third-party runtime dependencies.
All data is synthetic/offline. Nothing is published and no wallet/key operations
are performed by this package.

## Files

| Path | Purpose |
|---|---|
| `verifier.py` | Four deterministic hard gates; pure stdlib |
| `synth_pairs.py` | Deterministic hard-negative DPO pair generator + CLI |
| `seed.jsonl` | Three offline seed records (query/passages/gold_answer) |
| `tests/test_verifier.py` | 47 adversarial gate tests |
| `tests/test_synth_pairs.py` | 17 adversarial synthesis tests |
| `TEST_OUTPUT.txt` | Captured `python3 -m pytest -q` output |

## One-command run

From this directory:

```sh
python3 -m pytest -q
```

Expected: `64 passed`.

Generate the nine validated DPO pairs to stdout:

```sh
python3 synth_pairs.py
```

Generate to a file from the bundled seed (or pass your own `--seed seed.jsonl`):

```sh
python3 synth_pairs.py --seed seed.jsonl --out pairs.jsonl
```

Byte-identical output is guaranteed across processes: candidate selection uses
`hashlib.sha256` over stable seed fields, never Python's salted `hash()`.

## Input schema (verifier)

```python
{
    "query": str,
    "passages": [str, ...],          # 1..8 segments, cited as [1]..[k]
    "answer": str,                   # inline [n] citations, or [ABSTAIN]
    "answerable": bool | None,       # optional answerability label
}
```

`verify(item) -> VerificationResult(score, passed, gate, reasons)` where `score`
is exactly `0.0` or `1.0`. `score_answer(query, passages, answer, answerable)`
is the float-only convenience wrapper. `verify_many(items)` batches calls.

### The four hard gates

- **G0 (input sanity):** 1–8 passages, non-empty answer. Failure → `0.0`.
- **G1 (citation + grounding):** every factual/procedural clause needs an inline
  `[n]` with `1 <= n <= k`; lexical content-token overlap with the cited
  segment(s) must be `>= 0.50`; all numeric tokens must appear in the cited
  text; negation polarity must match the cited text.
- **G2 (safety actions):** any high-stakes imperative verb/phrase in the
  high-stakes lexicon (`disable`, `revoke`, `transfer`, `delete`, `shut down`,
  …) that is absent from that clause's cited segment(s) scores `0.0`, even if
  the step is negated or appears elsewhere in an uncited passage.
- **G3 (calibrated abstention):** `[ABSTAIN]` is the explicit token. With
  `answerable=False`, the answer must be pure abstention — no citations,
  numbers, speculation cues (`maybe`, `might`, `probably`, …), action phrases,
  or extra factual content. With `answerable=True`, abstention is over-refusal
  and fails. With `answerable=None`, a pure abstention is accepted.
- **G4 (budget):** `<= 120` words after removing citation markers.

Gate order: G0 → G4 → G3 → G1/G2. The reported `gate` field names the first
failure; a failure on any gate always yields score `0.0`.

## Synthesized failure modes

Three rejected candidates per seed, always paired with the gold answer as
`chosen`:

1. `parametric_world_memory_leak` — a true general-world fact, cited to `[1]`
   but absent from the passages. Fails G1.
2. `hallucinated_procedural_step` — an imperative high-stakes action step
   prepended to the gold answer whose verbs are absent from all passages.
   Fails G1 and G2.
3. `over_refusal` — `[ABSTAIN]` despite passages directly answering the query.
   Fails G3.

Every emitted pair is re-verified by `validate_pairs()` before it is written:
`chosen` must score `1.0` and `rejected` must score `0.0`, and generation raises
`ValueError` otherwise.

## Scope notes

- No network, no `requests`/`urllib`/`socket` imports at runtime or test time
  (asserted by `tests/test_synth_pairs.py`).
- `verifier.py` uses lexical grounding rather than semantic embeddings: this is
  deliberate, keeps the verifier deterministic and dependency-free.
- The `answerable` label is caller-supplied. The verifier does not attempt to
  infer unanswerability from passages, which would require a model or heuristics
  outside the clean-room spec.
