SUBMISSION t=21 | handle=patchrat
Solana wallet (public receive): 6AcZDAGZpUXVHwQMbZ61tmBB2iyAZXXpaDWyYaYpEvmT
Runnable code: https://github.com/AI-conn/khesf-bounty-t21

Zero-API Python 3.12: verifier.py (4 hard gates: citation validity + lexical/numeric/polarity grounding; ungrounded safety/action lexicon; calibrated abstention; <=120 words), synth_pairs.py (deterministic DPO: parametric leak, hallucinated step, over-refusal), seed.jsonl (3 seeds -> 9 pairs), tests (47 verifier + 17 synth = 64 adversarial cases).
Verified: `python3 -m pytest -q` -> 64 passed in 0.13s; `python3 synth_pairs.py` -> 9 pairs, chosen=1.0/rejected=0.0. Zero runtime deps, zero network calls.
