# Pilots 025–027: sparse and regression-aware contextual directions

Status: completed; 58 tests passed per job.

025 independently refits the contextual direction on three new practice
seeds. Test all 15 nonempty subsets of the four fixed rows at gains 1 and 2,
plus no-op (31 practice configurations). Select maximal raw accuracy subject
to zero or one paired protection losses, breaking ties by fewer losses,
fewer rows, then smaller gain. Independently evaluate those locked selections,
full support and the previously exploratory omit-0, omit-2 and single-1
supports. No held-out selection. Report every practice configuration.

026 uses weighted ridge: minimize squared desired raw shifts, weighted
squared protected shifts, and L2 regularization. Compare weights 1,10,100
on protected examples; rescale each direction to the ordinary ridge norm.
Fit on separate practice seeds and include alternate wording; choose gains
on distinct practice examples, not fitting examples or held-out data.
Compare fixed gain 2 and zero/one-loss selected variants against ordinary
ridge selected with zero losses. Norm matching isolates direction at fixed
gain, but this is a soft representation-level constraint, not guaranteed
behavior preservation. No runtime hooks or router.

027 decomposes raw JSON failures into correct raw, correct content wrapped,
raw wrong content, wrapped wrong content, and other invalid responses.
Compare familiar and schema templates over two fresh seeds, base, global
and fixed-gain contextual edits. Strict full-response parsing and a single
outer Markdown fence; no substring recovery. Retain the official exact
scorer and paired regression counts. Python object equality is a diagnostic
and can conflate booleans with integers. No selection based on these results.

CPU only, no paid APIs or GPU; five-hour job cap and partial artifacts.

## Verified outcomes

025: 31 practice configurations per seed, 93 total. Zero-budget supports [0,1] gain 2, [0] gain 1, [0,1,3] gain 2. One-budget supports [0,1,3], [0,1], [0,1,3], all gain 2. Held-out raw / paired protected losses: base 18/144 and 0; global 99 and 24; full 95 and 11; omit 0 54 and 3; omit 2 95 and 12; single 1 43 and 0; zero-budget 71 and 8; one-budget 92 and 10. Protected denominator 288, including baseline-incorrect examples; paired losses count correct-to-incorrect transitions only. Earlier notification quoted larger loss counts for several arms; these numbers explicitly sum fenced and other losses only. Exact per-group paired results are in JSON. All three fits found nonzero zero-budget patches but held-out preservation is not assured. Fixed single row index 1 is layer 16 channel 3532; future replication must use new data.

026: base 17/144 raw; global 95 with 28 protection losses; fixed gain ordinary/10/100 all 96 with 21/19/19 protection losses. Higher penalties reduced activation fitting MSE but neither chosen; zero-budget selected 54/144 with ten protection losses, one-budget 68 with 14. One fit chose no-op for either budget. Soft last-position proxy not behavioral guarantee.

027: familiar raw base 0/48, global and contextual 35/48. Base familiar categories 39 wrapper-only, eight wrapped wrong content, one invalid. New-schema base three correct wrapped, 33 wrapped wrong content, 12 raw wrong content; all arms zero correct raw. Across both kinds global 17 and contextual three paired protection losses. Underlying extraction remains a major limitation; no broad capability improvement established. Diagnostic equality may conflate bool/int.

[Exact log summaries](../results/sparse-contextual-025-027.json) · [CI artifacts](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45834707730).
