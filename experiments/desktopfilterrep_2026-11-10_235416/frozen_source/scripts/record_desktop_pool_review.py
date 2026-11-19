"""Persist the agent's training-only label review before the expanded family starts."""
import json
from reasoning_rl.data import load_split
from reasoning_rl.runtime import ROOT, sha256_file, json_write

rows = load_split("train", "develop", limit=160, seed=20261105)[128:]
excluded = {
    "gsm8k-train-05134": "Jared=0.85*Ann, but the reference uses Ann=1.15*Jared; percentage baseline contradicts the question.",
    "gsm8k-train-07439": "Question says 'left eaten' while the reference reports uneaten oranges; 18 eaten versus 78 uneaten is unresolved wording.",
    "gsm8k-train-02307": "Filling a sandbox requires depth/volume; the stated bag amount and reference use square feet only. Exclude dimensional ambiguity.",
    "gsm8k-train-01188": "The 10% gratuity base is not specified (food-only versus food plus delivery); do not reward one interpretation as uniquely correct.",
}
calculations = [
    ("01954", "12+12*6", "84"), ("01408", "26*(20+30), using the conventional 52-week year", "1300"),
    ("00597", "3*52+3*26-34", "200"), ("00154", "(400+200)*(1+2*0.10), simple interest", "720"),
    ("00174", "5*2+5*5", "35"), ("04291", "30+15*0.20", "33"),
    ("00638", "32*5+7*10+5000-200*2", "4830"), ("03208", "20+18+3*18", "92"),
    ("03888", "(20+30)*20", "1000"), ("06843", "(4-1)*12-6", "30"),
    ("06341", "21+4+7-2", "30"), ("03914", "10+2*3*10", "70"),
    ("05111", "(20+10)+(15+10)+(10+10)", "75"), ("06535", "112/14", "8"),
    ("04918", "14+7+11+22", "54"), ("04443", "3+9+18+3", "33"),
    ("02543", "4*5*7/35", "4"), ("06478", "2+4+16", "22"),
    ("03295", "(8+6)-16/2", "6"), ("06200", "2000-2000*3/5-300", "500"),
    ("07107", "3*(24+18)", "126"), ("03223", "9000*2/3", "6000"),
    ("07247", "2700/(1+0.5+3)*0.5", "300"), ("05497", "84/2*20/60", "14"),
    ("00753", "(29+17)*7", "322"), ("07359", "(110-40-30)/2", "20"),
    ("04025", "72/(3+5), assuming each consumed person is distinct as stated", "9"),
    ("06441", "(100-8*9)/2", "14"),
]
checked = {"gsm8k-train-"+suffix: (calculation, answer) for suffix, calculation, answer in calculations}
assert set(checked) | set(excluded) == {row["id"] for row in rows}
assert not set(checked) & set(excluded)
for row in rows:
    if row["id"] in checked:
        assert checked[row["id"]][1] == row["answer"]
path = ROOT / "experiments/desktop_expanded_train_review_20261105.json"
if path.exists():
    raise FileExistsError("Preserve the original review")
json_write(path, {
    "reviewer": "Codex agent", "human_reviewed": False, "scope": "training-only candidate positions 128:160",
    "data_seed": 20261105, "data_manifest_sha256": sha256_file(ROOT / "data/manifest.json"),
    "candidate_count": 32, "retained_count": 28,
    "rule": "Exclude explicit reference-condition contradictions, unresolved target wording, dimensional insufficiency, or unspecified percentage base; apply identically before all expanded C/D/E runs and all seeds.",
    "heldout_accessed": False, "based_on_expanded_model_outcomes": False,
    "raw_data_modified": False, "global_training_label_quality_claim": False,
    "items": [{"id": row["id"], "question_sha256": row["question_sha256"],
               "decision": "exclude" if row["id"] in excluded else "retain",
               "reason": excluded[row["id"]] if row["id"] in excluded else checked[row["id"]][0],
               "reference_answer": row["answer"]} for row in rows],
    "limitations": "Agent arithmetic and wording review of this 32-item GRPO pool only; conventional 52-week year retained; not human adjudication, not an audit of all SFT labels, not a claim of complete absence of ambiguity."
})
print(path)
