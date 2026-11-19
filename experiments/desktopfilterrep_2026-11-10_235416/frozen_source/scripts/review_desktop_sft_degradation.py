"""Record development-only observed failure patterns; never alter the frozen experiment."""
from collections import Counter
from reasoning_rl.runtime import ROOT, jsonl_read, json_write

prefix = "desktoprocm10pilot_20261105"
a = jsonl_read(ROOT / f"runs/{prefix}-A-val/validation/predictions.jsonl")
b = jsonl_read(ROOT / f"runs/{prefix}-B-val/validation/predictions.jsonl")
assert [r["id"] for r in a] == [r["id"] for r in b]
paired = Counter((x["correct"], y["correct"]) for x, y in zip(a, b, strict=True))
json_write(ROOT / f"reports/comparisons/{prefix}/sft_degradation_review.json", {
    "scope": "Fixed 32 development questions, after protocol design; no holdout access",
    "reviewer": "Codex agent", "human_reviewed": False,
    "baseline_correct": sum(r["correct"] for r in a), "sft_correct": sum(r["correct"] for r in b),
    "paired_counts": {f"A_{x}_B_{y}": count for (x, y), count in paired.items()},
    "sft_truncated_ids": [r["id"] for r in b if r["truncated"]],
    "sft_parse_error_ids": [r["id"] for r in b if not r["parse_ok"]],
    "sft_wrong_but_parseable_ids": [r["id"] for r in b if r["parse_ok"] and not r["correct"]],
    "observed_repetition_examples": [
        {"id": "gsm8k-train-00848", "observation": "Repeated '20 cows are on the farm' to the 512-token cap"},
        {"id": "gsm8k-train-05654", "observation": "Repeated 'So, the answer is 315' to the cap; no unambiguous final numeric answer under frozen scorer"},
        {"id": "gsm8k-train-05147", "observation": "Repeated the question about waitress wage and tips to the cap"}],
    "conclusion": "SFT loss reduction did not improve these development outcomes. Repetition/truncation and parseable wrong answers both contribute; this is not just strict #### format compliance.",
    "unconfirmed_hypotheses": ["Overfitting or optimization strength on the small SFT set", "Interaction between completion fine-tuning and generation/EOS behavior"],
    "causal_root_cause_confirmed": False, "configuration_or_checkpoint_changed_from_review": False,
    "next_decision": "Finish the frozen paired pilot and retain every endpoint; investigate any SFT repair in a new development family before final holdout authorization."
})
print("Saved development-only degradation review; no training configuration modified")
