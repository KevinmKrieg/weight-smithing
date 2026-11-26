# 028–030: single-neuron robustness and output-aware fitting
Status: submitted for CI; no results claimed.

028: three fresh fits of the predeclared layer-16/channel-3532 row (index 1
of JAC_PATCH), fixed gains 2 and 4, plus practice-selected gain among
0,1,2,3,4 under zero paired protection losses. Full contextual and global
comparators. Familiar, paraphrase and alternate prose templates; expanded
protected prompts include arithmetic, copying, ordering and classification.
Nine sets of 24 raw and 72 protected prompts (216 raw, 648 protected).
Zero observed losses are not a universal safety guarantee.

029: actual output-logit margin gradient with respect to the selected gate
row, computed by chain rule through all prompt/prefix positions. Fit positive
raw first-token margin changes versus strongest competing token and zero
change in protected top-two margins at up to three greedy prefix positions.
Weighted ridge penalties 1,10,100; norm match to the ordinary activation
direction, fixed gain 2 comparison, and separate practice gain selection.
Three independent fits; protected prefixes are frozen baseline prefixes,
not all possible generated sequences. Fit examples, practice selection
and held-out generation datasets are separate. Diagnostic gradient hooks
removed before evaluation; actual weight-only additions at inference.
First-order margin approximation is not a guaranteed behavioral constraint.
Gradient unit test compares against direct parameter autograd.

030: discover exact wrapper-only errors in CSV, integer, label, YAML and
identifier copy tasks; independent discovery and confirmation sets of 24
per family. Require at least three wrapper-only failures in each split.
No edit or held-out selection in this discovery pilot. Exact content
matching may undercount semantically valid alternatives.

CPU-only, no paid APIs or GPU. Five-hour job cap, partial artifact uploads.
