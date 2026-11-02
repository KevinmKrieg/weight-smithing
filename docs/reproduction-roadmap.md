# Reproduction roadmap

Source: https://rikiyat.github.io/blog/weight-space-programming/

## Completed: method-level partial reproduction of Experiment 5

- Exact MLP architecture 600 -> 32 tanh -> 32 tanh -> 6, including 20,486 weight/bias parameters.
- All-zero initialization and sparse, direct weight assignments for a horizontal intercept controller.
- Explicit zero-weight random-action baseline and two circuit variants, follow and lead.
- Reproducible paired episode evaluation on a small independent toy Breakout.
- Optional adapter to the upstream MinAtar Breakout environment (separately installed).
- Serialization, weight-inspection prompt, safe declarative LLM-patch input, plotting, tests and GitHub Actions CI.
- Initial offline benchmark on 300 toy episodes, summarized in results/toy-300-summary.json.

## Not reproduced

The author's model-produced weights, exact six-channel game interface, Seaquest and Asterix circuits, large PPO baselines, and all published numeric results. The direct-programmed controller is **human-designed**; it does not demonstrate that an LLM can write such weights.

The offline toy is NOT MinAtar. Even the optional upstream MinAtar adapter may differ from the published experiment's preprocessing/action semantics.

## Next steps

1. Validate channel semantics, episode handling, action interpretation and seed control against the original MinAtar code; run 1,000 episodes with sticky-action probability 0.1 and a longer horizon.
2. Quantify uncertainty with seeded paired comparisons, report individual scores and confidence intervals, and compare only policies evaluated in the same environment.
3. Train a fair PPO baseline in the exact matched harness, saving evaluation checkpoints and command lines.
4. Add Seaquest and Asterix, including the correct object channels and observation rules.
5. Run a genuinely one-shot LLM weight-writing trial. Record the prompt and JSON edit proposal **before** exposing any rollout scores. Never allow generated Python execution.
6. Attempt the article's Experiment 1 weight-reading task with task identities withheld and one uniform input encoding; quantify recognition accuracy.
7. Document limitations and replicate results across different seeds and models.

## Threats to validity

- The channel names and model interface partially disclose the task.
- Sticky actions and finite episodes vary results; report configuration and evaluation horizon.
- Our toy engine uses simplified collision physics and must not be used to claim the published benchmark.
- PPO numbers are incomparable unless trained and measured against the identical game code.
- No language model has generated weights in this initial release.
