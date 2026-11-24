"""Adversarial tests for deterministic DPO hard-negative synthesis."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from synth_pairs import (
    DEFAULT_SEED,
    FAILURE_MODES,
    load_seed,
    synthesize,
    synthesize_one,
    validate_pairs,
)
from verifier import _ACTION_VERBS, _WORD_RE, normalize, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pairs():
    return synthesize(DEFAULT_SEED)


def common_fields(pair):
    return {
        "query": pair["query"],
        "passages": pair["passages"],
        "answerable": pair["answerable"],
    }


def test_synth_default_seed_produces_nine_pairs(pairs):
    assert len(pairs) == 9


def test_synth_three_distinct_failure_modes(pairs):
    assert {p["failure_mode"] for p in pairs} == set(FAILURE_MODES)
    assert len(FAILURE_MODES) == 3


def test_synth_every_chosen_answer_passes_verifier(pairs):
    for pair in pairs:
        r = verify({**common_fields(pair), "answer": pair["chosen"]})
        assert r.passed, (pair["id"], r.reasons)


def test_synth_every_rejected_answer_fails_verifier(pairs):
    for pair in pairs:
        r = verify({**common_fields(pair), "answer": pair["rejected"]})
        assert r.score == 0.0, pair["id"]


def test_synth_recorded_scores_match_live_verification(pairs):
    for pair in pairs:
        assert pair["chosen_score"] == 1.0
        assert pair["rejected_score"] == 0.0


def test_synth_is_deterministic_within_process(pairs):
    assert synthesize(DEFAULT_SEED) == pairs


def test_synth_is_deterministic_across_processes():
    script = "import json,synth_pairs;print(json.dumps(synth_pairs.synthesize(synth_pairs.DEFAULT_SEED),sort_keys=True))"
    first = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert first == second


def test_synth_failure_mode_ids_are_unique(pairs):
    assert len({p["id"] for p in pairs}) == len(pairs)


def test_parametric_leak_fact_is_absent_from_all_passages(pairs):
    leaks = [p for p in pairs if p["failure_mode"] == "parametric_world_memory_leak"]
    assert len(leaks) == 3
    for pair in leaks:
        passage_text = normalize(" ".join(pair["passages"]))
        content = normalize(pair["rejected"])
        leak_tokens = _WORD_RE.findall(content)
        passage_tokens = set(_WORD_RE.findall(passage_text))
        # At least three content tokens of the leaked fact must be unsupported.
        unsupported = [t for t in leak_tokens if t not in passage_tokens]
        assert len(unsupported) >= 3, pair["id"]


def test_hallucinated_step_action_verb_is_absent_from_passages(pairs):
    steps = [p for p in pairs if p["failure_mode"] == "hallucinated_procedural_step"]
    assert len(steps) == 3
    for pair in steps:
        injected = pair["rejected"].split(". ")[0]
        verbs = _ACTION_VERBS.intersection(_WORD_RE.findall(normalize(injected)))
        assert verbs, pair["id"]
        passage_tokens = set(_WORD_RE.findall(normalize(" ".join(pair["passages"]))))
        assert not verbs.intersection(passage_tokens), pair["id"]


def test_over_refusal_rejected_is_pure_abstention(pairs):
    refusals = [p for p in pairs if p["failure_mode"] == "over_refusal"]
    assert len(refusals) == 3
    for pair in refusals:
        assert "[ABSTAIN]" in pair["rejected"]
        assert "[" not in pair["rejected"].replace("[ABSTAIN]", "")


def test_seed_jsonl_matches_builtin_seed():
    seeded = load_seed(ROOT / "seed.jsonl")
    assert seeded == [dict(s) for s in DEFAULT_SEED]


def test_validate_pairs_accepts_valid_pairs(pairs):
    validate_pairs(pairs)


def test_validate_pairs_rejects_tampered_pair(pairs):
    tampered = [dict(p) for p in pairs]
    tampered[0]["rejected"] = tampered[0]["chosen"]
    with pytest.raises(ValueError):
        validate_pairs(tampered)


def test_cli_writes_parseable_jsonl(tmp_path, pairs):
    out = tmp_path / "pairs.jsonl"
    proc = subprocess.run(
        [sys.executable, "synth_pairs.py", "--quiet", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout == ""
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert lines == pairs


def test_no_network_runtime_imports_in_sources():
    forbidden = ("import requests", "import urllib.request", "import socket", "from http.client")
    for name in ("verifier.py", "synth_pairs.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, (name, token)


def test_synthesize_one_rejects_broken_seed():
    broken = dict(DEFAULT_SEED[0])
    broken["gold_answer"] = "This is not grounded [1]."
    with pytest.raises(ValueError):
        synthesize_one(broken)
