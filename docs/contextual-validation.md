# Pilots 022–024: contextual causal validation
Status: submitted for CI; results pending.

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
