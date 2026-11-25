# Pilots 016–018: regression-aware editing

Status: completed; 51 tests passed per job.

016 supplies the pinned Qwen Coder editor with shared single-neuron calibration
on actual extraction practice tasks, continuous raw and required-fence margin
changes, per-prompt changes, and individual output examples from recent trials.
Compare six unique target trials per arm against greedy additive margin search
and one random search realization, with identical candidate and action spaces,
shared calibration cost, and selection rule. Search from best exploratory trial;
final patch requires an actual practice raw gain and zero paired practice
regression losses. Otherwise retain base. No held-out feedback. Report first
and selected patches and fixed Jacobian baseline. The editor sees only four
example outputs from each of its two most recent trials to bound context; all
examples are saved. This is not an unrestricted editor or powered superiority test.

017 screens the full 3^4 grid of attenuation values 0, 0.5, 1 on the fixed
Jacobian support by measured practice margin. Shortlist top four for each
penalty 0, 1, 4 and include no-op and full ablation. Measure actual generation
on 12 paired extraction and 12 unrelated practice examples. Save the
nondominated accuracy/loss frontier among evaluated candidates; this is not a
global frontier over all weight edits. Lock zero-loss and at-most-one-loss
practice selections before two new held-out seeds, with full ablation and base
controls. Practice guarantees do not imply held-out guarantees. No optimizer
or gradient-based weight updates.

018 is discovery only, not intervention. Screen two SmolLM2 checkpoints on
five exact-output task families (JSON extraction, label, integer, uppercase,
CSV). Define eligible wrapper-only errors before execution using exact scoring
after removal of one known wrapper. Require at least 3/18 discovery failures
and independently at least 3/24 confirmation failures before identifying an
eligible future target. Record every screened model/family including failures
and revisions. No edits or claims of cross-architecture transfer.

All CPU jobs have 180-minute limits and preserve raw artifacts. No paid API
or GPU requested. Benchmarks remain synthetic.

## Verified outcomes

016: editor/greedy/random selected patches each 4/24 raw vs 0/24 base, paired CI [4.17,33.33] pp. Editor first 1/24, greedy first 3/24, random first 0/24. Editor and greedy each lost one required fence; random none. Jacobian 18/24 raw with two fence losses. No demonstrated editor superiority. One realization only.

017: 81 margin-screened, ten generation-tested. Practice nondominated points were base (0 raw,0 losses) and partial patch (8 raw,2 losses), with layer 12 channel 765 at 0.5 and other Jacobian columns at zero. Both predeclared constrained selectors retained base. Full Jacobian held-out raw/fenced/other 36/34/19 vs 0/40/18, 37 wins and six losses, 89/144 vs 58/144. Partial frontier patch was not held-out tested.

018: 0/180 eligible wrapper-only failures, no confirmation triggered, no intervention. This does not establish absence of other repairable errors.

[Exact summaries](../results/regression-editor-016-018.json) · [Raw artifacts](https://github.com/KevinmKrieg/weight-smithing/actions/runs/45699944755).
