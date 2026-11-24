"""Adversarial tests for the four deterministic hard gates in verifier.py."""

import pytest

from verifier import MAX_WORDS, VerificationResult, score_answer, verify

LEDGER = "Node operators should prune old ledger archives when disk usage exceeds 90 percent."
WEATHER = "The forecast calls for heavy rain and strong coastal winds on Tuesday."


def item(answer, passages=None, answerable=None):
    return {
        "query": "test query",
        "passages": list(passages) if passages is not None else [LEDGER],
        "answer": answer,
        "answerable": answerable,
    }


def result(answer, passages=None, answerable=None):
    return verify(item(answer, passages, answerable))


# --------------------------------------------------------------------------
# G1 -- citation validity, lexical grounding, numeric + polarity alignment
# --------------------------------------------------------------------------

def test_g1_single_valid_citation_passes():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent [1].")
    assert r.passed and r.score == 1.0


def test_g1_answer_without_any_citation_fails():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent.")
    assert r.score == 0.0 and "G1" in r.gate


def test_g1_zero_index_is_out_of_range():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent [0].")
    assert r.score == 0.0


def test_g1_index_beyond_passage_count_fails():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent [2].")
    assert r.score == 0.0


def test_g1_absurdly_large_index_fails():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent [9999].")
    assert r.score == 0.0


def test_g1_uncited_second_clause_fails():
    r = result(
        "Prune old ledger archives when disk usage exceeds 90 percent [1]. "
        "Snapshot creation pauses automatically when storage is exhausted."
    )
    assert r.score == 0.0
    assert any("uncited" in reason for reason in r.reasons)


def test_g1_citation_to_unrelated_passage_fails_grounding():
    r = result(
        "Prune old ledger archives when disk usage exceeds 90 percent [2].",
        [LEDGER, WEATHER],
    )
    assert r.score == 0.0


def test_g1_correct_second_passage_citation_passes():
    r = result(
        "Prune old ledger archives when disk usage exceeds 90 percent [1]. "
        "Heavy rain and coastal winds are forecast [2].",
        [LEDGER, WEATHER],
    )
    assert r.passed


def test_g1_numeric_mismatch_fails():
    r = result("Disk usage exceeds 80 percent [1].")
    assert r.score == 0.0
    assert any("numeric" in reason for reason in r.reasons)


def test_g1_numeric_match_passes():
    r = result("Disk usage exceeds 90 percent [1].")
    assert r.passed


def test_g1_polarity_flip_fails():
    r = result(
        "Do not prune old archives [1].",
        ["Do not prune old archives."],
    )
    # Should pass: negation is aligned.  Now flip it below.
    assert r.passed
    flipped = result("Do not prune old archives [1].", ["Prune old archives."])
    assert flipped.score == 0.0
    assert any("polarity" in reason for reason in flipped.reasons)


def test_g1_malformed_bracket_is_not_a_citation():
    r = result("Prune old ledger archives when disk usage exceeds 90 percent [abc].")
    assert r.score == 0.0


def test_g1_world_memory_leak_is_not_grounded():
    r = result("The Eiffel Tower is located in Paris [1].")
    assert r.score == 0.0
    assert any("lexical grounding" in reason for reason in r.reasons)


def test_g1_duplicate_citations_are_harmless():
    r = result("Prune old ledger archives [1][1].")
    assert r.passed


def test_g1_high_partial_overlap_passes():
    r = result("Operators prune old ledger archives [1].")
    assert r.passed


# --------------------------------------------------------------------------
# G2 -- zero ungrounded imperative safety / procedural action phrases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("verb", ["disable", "revoke", "transfer", "delete", "override"])
def test_g2_ungrounded_imperative_verb_scores_zero(verb):
    r = result(f"{verb.capitalize()} the service [1].", ["The service is healthy."])
    assert r.score == 0.0
    assert any("G2" in reason for reason in r.reasons)


def test_g2_grounded_imperative_verb_passes():
    r = result(
        "Disable the service during maintenance [1].",
        ["Operators must disable the service during maintenance."],
    )
    assert r.passed


def test_g2_ungrounded_multiword_action_phrase_fails():
    r = result("Shut down the database [1].", ["The database is online."])
    assert r.score == 0.0
    assert any("shut down" in reason for reason in r.reasons)


def test_g2_action_present_only_in_uncited_passage_still_fails():
    r = result(
        "Revoke the credentials [1].",
        ["The credentials are active.", "Administrators may revoke the credentials."],
    )
    assert r.score == 0.0


def test_g2_negated_action_still_requires_grounding():
    r = result("Do not revoke the credentials [1].", ["The credentials are active."])
    assert r.score == 0.0


def test_g2_hallucinated_step_prepended_to_good_answer_fails():
    good = "Prune old ledger archives when disk usage exceeds 90 percent [1]."
    rejected = "First, delete every backup [1]. " + good
    assert result(good).passed
    assert result(rejected).score == 0.0


# --------------------------------------------------------------------------
# G3 -- calibrated abstention, no speculation, no over-refusal
# --------------------------------------------------------------------------

def test_g3_pure_abstention_on_unanswerable_query_passes():
    r = result("[ABSTAIN]", ["The service is healthy."], answerable=False)
    assert r.passed and r.gate == "G3"


def test_g3_verbose_abstention_phrasing_passes():
    r = result("Insufficient evidence [ABSTAIN]", ["The service is healthy."], answerable=False)
    assert r.passed


def test_g3_idontknow_abstention_passes():
    r = result("I don't know [ABSTAIN]", ["The service is healthy."], answerable=False)
    assert r.passed


def test_g3_abstention_with_citation_fails():
    r = result("[ABSTAIN] [1]", ["The service is healthy."], answerable=False)
    assert r.score == 0.0


def test_g3_partial_speculation_with_abstention_fails():
    r = result(
        "It might be the maintenance window [ABSTAIN]",
        ["The service is healthy."],
        answerable=False,
    )
    assert r.score == 0.0
    assert any("speculation" in reason for reason in r.reasons)


def test_g3_partial_guess_and_abstention_fails():
    r = result(
        "Possibly 90 percent [ABSTAIN]",
        ["The service is healthy."],
        answerable=False,
    )
    assert r.score == 0.0


def test_g3_numeric_claim_inside_abstention_fails():
    r = result("Insufficient data about 48 hours [ABSTAIN]", ["The service is healthy."], answerable=False)
    assert r.score == 0.0


def test_g3_factual_content_inside_abstention_fails():
    r = result("The Eiffel Tower is in Paris [ABSTAIN]", [LEDGER], answerable=False)
    assert r.score == 0.0


def test_g3_over_refusal_on_answerable_query_fails():
    r = result("[ABSTAIN]", [LEDGER], answerable=True)
    assert r.score == 0.0
    assert any("over-refusal" in reason for reason in r.reasons)


def test_g3_plain_idontknow_without_token_is_not_abstention():
    r = result("I don't know.", [LEDGER], answerable=False)
    assert r.score == 0.0


def test_g3_pure_abstention_accepted_when_label_unknown():
    r = result("[ABSTAIN]", [LEDGER], answerable=None)
    assert r.passed


# --------------------------------------------------------------------------
# G4 -- strict word budget and input sanity
# --------------------------------------------------------------------------

def test_g4_exactly_cap_words_passes():
    answer = " ".join(["prune"] * MAX_WORDS) + " [1]"
    assert result(answer).passed


def test_g4_one_word_over_cap_fails():
    answer = " ".join(["prune"] * (MAX_WORDS + 1)) + " [1]"
    r = result(answer)
    assert r.score == 0.0 and "G4" in r.gate


def test_g4_citation_markers_do_not_count_as_words():
    answer = " ".join(["prune"] * MAX_WORDS) + " " + "[1] " * 50
    assert result(answer).passed


def test_g4_empty_answer_fails():
    assert result("").score == 0.0


def test_g0_zero_passages_fails():
    assert result("Prune archives [1].", []).score == 0.0


def test_g0_nine_passages_fails():
    assert result("Prune archives [1].", [LEDGER] * 9).score == 0.0


def test_g0_eight_passages_is_valid_boundary():
    r = result("Prune old ledger archives [8].", [WEATHER] * 7 + [LEDGER])
    assert r.passed


# --------------------------------------------------------------------------
# Meta properties
# --------------------------------------------------------------------------

def test_scores_are_always_exactly_zero_or_one():
    answers = [
        "Prune old ledger archives [1].",
        "The Eiffel Tower is in Paris [1].",
        "[ABSTAIN]",
        "Disable everything [1].",
        "",
    ]
    for answer in answers:
        for label in (None, True, False):
            assert verify(item(answer, answerable=label)).score in (0.0, 1.0)


def test_verification_is_repeatable_and_pure():
    payload = item("Prune old ledger archives when disk usage exceeds 90 percent [1].")
    first = verify(payload)
    second = verify(payload)
    assert first == second
    assert payload["answer"] == "Prune old ledger archives when disk usage exceeds 90 percent [1]."


def test_result_is_a_frozen_binary_dataclass():
    r = verify(item("[ABSTAIN]", answerable=False))
    assert isinstance(r, VerificationResult)
    assert bool(r) is True
    with pytest.raises(Exception):
        r.score = 0.5


def test_score_answer_convenience_matches_verify():
    direct = verify(item("Prune old ledger archives when disk usage exceeds 90 percent [1]."))
    wrapper = score_answer("q", [LEDGER], "Prune old ledger archives when disk usage exceeds 90 percent [1].")
    assert wrapper == direct.score
