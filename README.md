<div align="center">
  <img src="assets/llm-weight-editor.svg" width="100%" alt="One LLM directly edits another LLM's neural weights" />
</div>

# Weight Smithing

**Can an LLM improve another LLM by editing its weights directly — without gradient descent?**

An independent experiment: give an editor LLM a target model's **actual MLP weight summaries, practice-only activations and failures**, then ask it to propose sparse changes to its parameters. Evaluate the original and modified models on **held-out prompts**, alongside matched random-edit and activation-ranked controls.

Inspired by [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/), but with a **different target model, benchmark and editing protocol**. We are not chasing the original paper's reported scores.

## Experiment 01: Instruction following through direct weight edits

| Component | Choice |
|---|---|
| Target | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), ~0.49B parameters |
| Editor | Configurable LLM API, default gpt-4.1; or a separately recorded patch |
| Benchmark | Our own exact JSON, integer arithmetic, uppercase single-word and alphabetical-selection tasks |
| Intervention | Multiply up to 8 chosen internal MLP down-projection weight columns by bounded factors |
| Gradient updates | **Zero**: no optimizer, LoRA or fine-tuning |
| Controls | Original, editor patch, activation-ranked non-LLM patch, matched random patch |
| Metrics | Held-out exact match, paired bootstrap 95% CI, token use, repeats, per-family accuracy |

### Workflow

~~~mermaid
flowchart LR
  A["Qwen2.5-0.5B"] --> B["Practice prompts + failures"]
  B --> C["Read activation and weight statistics"]
  C --> D["Editor LLM proposes sparse JSON patch"]
  D --> E["Modify actual MLP weight columns"]
  E --> F["Held-out benchmark vs controls"]
~~~

An LLM can choose only from **practice-derived** candidate channels, based on their failure/success activation contrast and actual projection-weight fingerprints. A strict validator checks every requested modification. The proposed patch is **saved before any held-out evaluation**. It changes real model parameters, not the system prompt or a runtime activation hook.

### Current results

**No verified Qwen improvement yet.** Offline tests validate the experiment machinery; the next phase is a model run and the first LLM-authored patch. Any success or failure will be published with full per-example results. We won't substitute the original paper's results for our own.

## Run it

Python 3.10+, access to the Qwen checkpoint, and an editor API key for the full LLM-authored workflow:

~~~bash
git clone https://github.com/KevinmKrieg/weight-smithing.git
cd weight-smithing
python -m pip install -e '.[target,editor,dev]'
pytest -q

# Practice only — produces diagnostics from real Qwen weights.
python -m weightsmith.practice --per-family 4 --out artifacts/practice-001

# Editor LLM creates a patch before held-out testing. API usage may incur costs.
export OPENAI_API_KEY=your_key_in_your_shell
python -m weightsmith.experiment --editor openai --editor-model gpt-4.1 --out artifacts/qwen-001

# Plot only results produced by the experiment.
python -m weightsmith.plot --run artifacts/qwen-001 --out artifacts/qwen-001/charts
~~~

There is also a fully offline **non-LLM control** mode, using --editor heuristic, and a manually supplied patch mode using --editor file --proposal-file proposal.json. Do **not** label either mode LLM-generated unless an LLM actually authored the proposal.

The first automated practice-only GitHub Actions workflow deliberately avoids held-out evaluations. Full methodology and safeguards: [research protocol](docs/research-protocol.md). See the [research roadmap](docs/reproduction-roadmap.md).

## Supporting experiment

Earlier we built a tiny **hand-programmed** neural controller and toy Breakout experiment, with real charts and results. It demonstrates sparse weight editing but **not one LLM editing another LLM**. It remains available as [the Breakout experiment archive](docs/breakout.md).

## Milestones

| Stage | Status |
|---|---|
| Custom benchmark + practice/held-out separation | Implemented, unit-tested |
| Direct tensor patch + guaranteed restoration | Implemented, unit-tested |
| MLP activation and weight diagnostics | Implemented, unit-tested |
| LLM editor, matched controls and charts generator | Implemented, not yet run on Qwen |
| Actual Qwen practice diagnostics | Pending run |
| Actual LLM-generated patch and held-out metrics | **Pending** |

We will publish **real before/after metrics and interesting examples** when available, including negative results.

---

<sub>Independent MIT-licensed research project. No claimed affiliation with the source authors or model publishers.</sub>
