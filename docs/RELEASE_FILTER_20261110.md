# Fixed S2 prefilter vs full-pool uniform: three paired development seeds

`desktopfilter_20261110` compares uniform GRPO over the same fixed 64-question candidate pool (U) with uniform GRPO over the 16 questions retained by a fixed S2/K=4 stochastic probe (F). Retention uses raw success counts 1, 2 or 3. Model, S2, reward, generation and optimization settings are shared. Seeds are 13, 29 and 47.

|seed|U correct /128|F correct /128|F−U percentage points|
|---|---:|---:|---:|
|13|100|102|+1.5625|
|29|101|102|+0.78125|
|47|100|103|+2.34375|

Mean development accuracy is 78.385% / 79.948% (+1.5625 percentage points). Zero-variance rollout groups fall from 81.109% to 51.876%; effective advantage completions per charged end-to-end second rise from 0.10233 to 0.17646 (+72.45%). Each F endpoint pays the full 292.832841-second initial probe, including when reusing its frozen outputs. All six independent checkpoint-reload evaluations are charged; all endpoints remain below the common 1830-second cap.

This is a **development prerelease**, not evidence of stable generalization. The reused confirmation128 is development data; the conditional question bootstrap interval crosses zero. GSM8K test and SVAMP remain sealed. A training question with self-referential wording was identified after freezing and is disclosed with its signal contribution. The interrupted original U/13 attempt, its partial outputs and cost uncertainty are retained; the main six completed endpoints use the `-r1-` prefix with the original frozen parameters and budgets.

The complete Chinese report is [findings.md](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/fixed-prefilter-dev-20261110/reports/comparisons/desktopfilter_20261110/findings.md). [Reproduction commands](https://github.com/INeedSomeSugar/reasoning-rl-lab/blob/fixed-prefilter-dev-20261110/docs/REPRODUCING_FILTER_20261110.md), frozen source/data IDs, raw predictions/rewards, masks, gradients, actual parameter changes and cost ledgers are committed. The existing 59 regression tests and evidence audit passed; packaging rechecks all 2536 closeout-manifest files and the frozen execution source, data and starting weights.

Assets:

- `desktopfilter_20261110-source-and-evidence.zip`: exact bytes from the release's code commit for the declared portable source/evidence subset. Formal holdout contents and binary weights are excluded. The original pre-publication closeout snapshot is preserved; later publication metadata is on main.
- `desktopfilter_20261110-lora-and-optimizer-diagnostics.zip`: **nine adapters**, comprising six paired training endpoints, one common S2 start, one calibration endpoint and one save/reload diagnostic. Seven Adam/RNG diagnostic states are included. Automatic exact sampler/dataloader resume is not implemented.
- `archive_receipt.json`: source commit, archive SHA256/size, checkpoint roles and verification details.
- `SHA256SUMS.txt`: checksums for both ZIPs and the receipt.

The base model, tokenizer copies, virtual environment and caches are not included. Use the pinned Qwen/Qwen3-1.7B revision in the protocol and base tokenizer. No cloud GPU or paid API was used. Publication/upload time is separate from the original four-hour model-and-handoff budget.
