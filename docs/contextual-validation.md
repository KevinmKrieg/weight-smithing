# Pilots 022–024: contextual causal validation
Status: completed; 55 tests passed per job.

022 refits the Pilot 020 procedure on three independent practice seeds, with
12 raw/fenced/other examples each. Select from five gains with zero paired
practice regression losses; ties select least gain. Lock directions and gain
before evaluating familiar, paraphrased and new schema sets (24 examples per
group per set; 648 total evaluation prompts across three fits). Compare base
and global attenuation. This replicates the procedure, not the exact original
patch; synthetic templates shared across seeds are not independent domains.

023 fits one true direction per layer and three shuffled-label and three
isotropic random controls, matching direction norm per layer and thus row
perturbation norms with identical supports and gain 2 (predeclared from 020).
The true selected gain is also reported. Shuffling is across all practice
examples, preserving class counts. Evaluate single-row and leave-one-row-out
variants to investigate contributions and interactions, not norm-matched
full-patch controls. Two fresh held-out sets; no control selection using
held-out data. All locked vectors and raw responses preserved. Three
realizations of each null are descriptive, not a formal randomization test.

024 tests a distinct CSV wrapper failure on the same target. Require at least
three exact wrapper-only failures in 18 practice examples; otherwise report
infeasibility without editing. Fit directions on CSV raw vs fenced/unrelated
practice representations with fixed JSON-derived neuron support; choose gain
using the same zero-loss rule. Evaluate fresh CSV and JSON preservation
examples. Exact-string CSV scoring is intentionally strict; this is not
cross-architecture transfer or a general-capability benchmark.

CPU-only, no paid API or GPU. Four-hour job caps; partial artifacts uploaded.

## Verified outcomes

022: gains 0,0,2. At gain 2 practice raw 10,9,10 of 12; respectively one,one,zero regression losses. Third fit familiar 18/24 vs 0, paraphrase 14/24 vs 6, schema 0/24 vs 0. One contextual paired loss vs ten global in that fit. All 72 new-schema raw prompts failed under base, global and selected contextual variants. Across all 216 raw prompts selected contextual 44/216 vs base 18/216, global 96/216; two no-op selections are retained in aggregate. No claim of schema transfer.

023: selected gain 1, 25/48 raw with three paired fence losses. Fixed gain 2, 32/48 with three losses vs global 33/48 with eight paired losses. Norm-matched random 6,7,6; shuffled 10,7,7; base 6/48. Single rows 0–3 raw 10,15,6,10; leaveout rows 0–3 raw 18,16,32,30. Leaveout 0 and single 1 zero losses; leaveout 2 preserves full measured accuracy with three losses. These are exploratory held-out observations; future tests must use fresh data. Three null draws each are descriptive, not a significance test.

024: zero eligible CSV wrapper failures in 18 practice prompts; no intervention or JSON preservation evaluation. Cannot conclude behavioral transfer failed, only feasibility threshold was unmet.

[Exact summaries](../results/contextual-validation-022-024.json) · [Raw artifacts](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45798999117).
