"""Deterministic hard-negative DPO pair synthesizer (bounty t=21, zero API).

Given a seed JSONL of ``(id, query, passages, gold_answer)`` records, synthesize
exactly three rejected candidates per seed:

  * ``parametric_world_memory_leak`` -- true in general knowledge, absent here
  * ``hallucinated_procedural_step`` -- imperative safety step absent from text
  * ``over_refusal`` -- abstention despite directly answerable passages

Every candidate is checked with :func:`verifier.verify`: the chosen answer must
score 1.0 and the rejected answer 0.0.  Selection uses SHA-256 of stable seed
fields, never Python's salted ``hash()``, so output is byte-identical across
processes and machines.  No network calls, no randomness, stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from verifier import _ACTION_VERBS, _WORD_RE, normalize, verify

__all__ = [
    "FAILURE_MODES",
    "DEFAULT_SEED",
    "synthesize",
    "synthesize_one",
    "load_seed",
    "write_pairs",
    "validate_pairs",
    "main",
]

FAILURE_MODES = (
    "parametric_world_memory_leak",
    "hallucinated_procedural_step",
    "over_refusal",
)

# Facts that are true in general knowledge but deliberately absent from the
# synthetic passages below.  They are cited to [1] to simulate false grounding.
_WORLD_FACTS = (
    "The Eiffel Tower is located in Paris [1].",
    "Water boils at 100 degrees Celsius at sea level [1].",
    "Python was created by Guido van Rossum [1].",
    "A leap year contains 366 days [1].",
    "The Amazon River flows through Brazil [1].",
)

# High-stakes imperative steps that must never be invented from silence.
_INJECTED_STEPS = (
    "First, revoke all credentials and delete every backup [1].",
    "First, disable the monitoring agent and bypass the approval gate [1].",
    "First, transfer the treasury funds to an external wallet [1].",
    "First, shut down the database and wipe the audit log [1].",
    "First, override the safety interlock and skip verification [1].",
)

DEFAULT_SEED: tuple[dict[str, Any], ...] = (
    {
        "id": "seed-01-disk",
        "query": "What should an operator do when the node's disk is full?",
        "passages": [
            "Node operators should prune old ledger archives when disk usage exceeds 90 percent.",
            "The node automatically pauses snapshot creation when storage is exhausted.",
        ],
        "gold_answer": (
            "Prune old ledger archives when disk usage exceeds 90 percent [1]. "
            "Snapshot creation pauses automatically when storage is exhausted [2]."
        ),
    },
    {
        "id": "seed-02-multisig",
        "query": "How is the treasury multisig configured?",
        "passages": [
            "The treasury multisig requires three of five signers.",
            "Signers must use hardware wallets and store seed phrases offline.",
            "The timelock delay is 48 hours.",
        ],
        "gold_answer": (
            "The treasury multisig requires three of five signers [1]. "
            "Signers must use hardware wallets and store seed phrases offline [2]. "
            "The timelock delay is 48 hours [3]."
        ),
    },
    {
        "id": "seed-03-severity",
        "query": "What is the incident severity policy?",
        "passages": [
            "Severity one incidents require paging the on-call engineer within five minutes.",
            "Severity two incidents are handled during business hours.",
            "All incidents must be recorded in the incident log.",
        ],
        "gold_answer": (
            "Severity one incidents require paging the on-call engineer within five minutes [1]. "
            "Severity two incidents are handled during business hours [2]. "
            "All incidents must be recorded in the incident log [3]."
        ),
    },
)


def _pick(options: Sequence[str], seed_id: str, query: str, domain: str) -> str:
    digest = hashlib.sha256(f"{domain}\0{seed_id}\0{query}".encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _ungrounded_step(seed_id: str, query: str, passages: Sequence[str]) -> str:
    """Pick an injected step whose action verbs are absent from all passages."""
    passage_tokens = set(_WORD_RE.findall(normalize(" ".join(passages))))
    digest = hashlib.sha256(f"step\0{seed_id}\0{query}".encode("utf-8")).digest()
    order = [(_INJECTED_STEPS[(digest[0] + i) % len(_INJECTED_STEPS)]) for i in range(len(_INJECTED_STEPS))]
    for candidate in order:
        verbs = _ACTION_VERBS.intersection(_WORD_RE.findall(normalize(candidate)))
        if not verbs.intersection(passage_tokens):
            return candidate
    # Deterministic fallback: a verb that cannot occur in any real passage.
    return "First, frobnicate the safety enclave and purge the trust store [1]."


def synthesize_one(seed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the three deterministic DPO pairs for a single seed record."""
    seed_id = str(seed["id"])
    query = str(seed["query"])
    passages = [str(p) for p in seed["passages"]]
    gold = str(seed["gold_answer"])

    chosen_item = {
        "query": query,
        "passages": passages,
        "answer": gold,
        "answerable": True,
    }
    chosen_result = verify(chosen_item)
    if not chosen_result.passed:
        raise ValueError(f"gold answer for {seed_id} fails verifier: {chosen_result.reasons}")

    rejected = {
        "parametric_world_memory_leak": _pick(_WORLD_FACTS, seed_id, query, "fact"),
        "hallucinated_procedural_step": _ungrounded_step(seed_id, query, passages) + " " + gold,
        "over_refusal": "[ABSTAIN]",
    }

    pairs: list[dict[str, Any]] = []
    for mode in FAILURE_MODES:
        rejected_item = {
            "query": query,
            "passages": passages,
            "answer": rejected[mode],
            "answerable": True,
        }
        result = verify(rejected_item)
        if result.passed:
            raise ValueError(f"rejected pair {seed_id}/{mode} unexpectedly passed verifier")
        pairs.append(
            {
                "id": f"{seed_id}--{mode}",
                "query": query,
                "passages": passages,
                "answerable": True,
                "chosen": gold,
                "rejected": rejected[mode],
                "failure_mode": mode,
                "chosen_score": 1.0,
                "rejected_score": 0.0,
            }
        )
    return pairs


def synthesize(seeds: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Synthesize all pairs for an iterable of seed records."""
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        pairs.extend(synthesize_one(seed))
    return pairs


def load_seed(path: str | Path) -> list[dict[str, Any]]:
    """Read seed JSONL.  Blank lines are ignored."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("id", "query", "passages", "gold_answer"):
                if key not in record:
                    raise ValueError(f"line {line_number}: missing {key!r}")
            records.append(record)
    return records


def validate_pairs(pairs: Sequence[Mapping[str, Any]]) -> None:
    """Re-verify every pair; raise on any chosen/rejected score mismatch."""
    for pair in pairs:
        common = {
            "query": pair["query"],
            "passages": pair["passages"],
            "answerable": pair.get("answerable", True),
        }
        chosen = verify({**common, "answer": pair["chosen"]})
        rejected = verify({**common, "answer": pair["rejected"]})
        if chosen.score != 1.0 or rejected.score != 0.0:
            raise ValueError(
                f"invalid pair {pair.get('id')}: chosen={chosen.score} rejected={rejected.score}"
            )


def write_pairs(pairs: Iterable[Mapping[str, Any]], out: str | Path | None = None) -> None:
    """Write JSONL to ``out`` or stdout."""
    stream = Path(out).open("w", encoding="utf-8") if out else sys.stdout
    try:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if out:
            stream.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", help="seed JSONL path (default: built-in 3-record seed)")
    parser.add_argument("--out", help="output JSONL path (default: stdout)")
    parser.add_argument("--quiet", action="store_true", help="suppress the stderr summary")
    args = parser.parse_args(argv)

    seeds = load_seed(args.seed) if args.seed else list(DEFAULT_SEED)
    pairs = synthesize(seeds)
    validate_pairs(pairs)
    write_pairs(pairs, args.out)
    if not args.quiet:
        print(f"synthesized {len(pairs)} validated pairs from {len(seeds)} seeds", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
