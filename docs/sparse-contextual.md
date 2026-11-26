# Pilots 025–027: sparse and regression-aware contextual directions

Status: submitted for CI; no results claimed.

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
