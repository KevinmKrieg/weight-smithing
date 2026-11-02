# Weight Smithing

**Programming neural networks directly in weight space — no gradients, no training.**

An independent, auditable partial reproduction of the ideas in [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, November 2026).

## Scope

This repository demonstrates the central technique of the article's Experiment 5: constructing a small game-playing neural policy by **writing individual weights**, without optimization. It uses the reported **600 → 32 tanh → 32 tanh → 6** architecture (20,486 trainable parameters counting biases).

It is a **method-level partial reproduction**, not the authors' generated weights, exact harness, full game suite, trained PPO baseline, or reported results. The current networks are **hand-designed**, *not authored by an LLM*. No evidence is claimed for the article's other experiments.

Three compared policies:
- **zero:** all-zero MLP, sampling uniformly among the six actions.
- **follow:** sparse direct-write circuit measuring ball X minus paddle X.
- **lead:** same circuit, adding two times the difference between current and previous ball X (a velocity proxy).

The evaluation uses a small independent 10x10 toy Breakout (offline smoke test) or an optional adapter to the separately installed official MinAtar Breakout implementation. Different environment details can change reward substantially; do not compare these scores to the article's PPO benchmarks.

## Reproduced toy-environment results

300 seeded episodes per policy, 500-step maximum, 0.1 sticky-action probability, temperature 1:

| Policy | Mean reward | Sample SD |
|---|---:|---:|
| Zero weight random baseline | 1.547 | 0.827 |
| Direct-written follow | 1.913 | 1.171 |
| Direct-written lead | **7.820** | 8.136 |

The lead circuit edits **318 / 20,486** parameters (1.55%); mean paired improvement over zero is 6.273 with standard error 0.476. These results were obtained on the *independent toy* environment only. Machine-readable results are in [results/toy-300-summary.json](results/toy-300-summary.json).

## Quick start

Python 3.10+:

~~~bash
git clone https://github.com/KevinmKrieg/weight-smithing.git
cd weight-smithing
python -m pip install -e '.[dev]'
pytest -q
python -m weightspace.evaluate --backend toy --episodes 300 --out artifacts/toy-results.json
~~~

To exercise the optional original MinAtar Breakout API (not equivalent to the published evaluation harness):

~~~bash
python -m pip install -e '.[minatar]'
python -m weightspace.evaluate --backend minatar --episodes 100 --max-steps 1000 --out artifacts/minatar-results.json
~~~

The native MinAtar actions are **no-op, left, up, right, down, fire**. Its four Breakout channels are remapped into the generic six channels (unused channels set to zero).

Inspect and save the weight circuit, including an LLM-friendly *weight-only* prompt:

~~~bash
python -m weightspace.inspect --policy lead --export artifacts/lead.npz --prompt artifacts/lead-prompt.txt
~~~

Generate a benchmark plot from a saved full-results JSON:

~~~bash
python -m weightspace.plot --results artifacts/toy-results.json --out artifacts/toy-comparison.png
~~~

A separate **LLM-proposed edit** route is implemented as safe, declarative JSON; see [the one-shot challenge](docs/llm-proposal-prompt.md). A language model has not yet generated or evaluated a proposal, and the patch handler never executes generated code.

## How direct weight writing works

1. Start with every weight and bias initialized to exactly zero.
2. In first-layer neuron 0, place spatial coefficients that project the observed ball and paddle locations into their horizontal difference.
3. In the lead version, first-layer neuron 1 estimates horizontal ball displacement from the previous frame.
4. Pass the values through the second tanh layer and hand-set output weights so positive offsets favor right, negative offsets favor left, and near-zero offsets favor no-op.

The entire circuit is auditable in [src/weightspace/program.py](src/weightspace/program.py). No loss function, backpropagation, training data, optimizer, or fitting call is present.

## Research roadmap

See [reproduction roadmap](docs/reproduction-roadmap.md) for official-environment evaluation, same-environment PPO controls, additional games, LLM-written tensors, and blind weight-reading evaluation.

## Attribution / license

Inspired by Takehi, Teoh, Zhang, and Khattab. This implementation was written independently and makes no claim of affiliation or authorship of the original experiments. MinAtar is an optional external dependency with its own license. This repository is MIT-licensed.
