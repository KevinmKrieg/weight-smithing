# Reproduction roadmap

Reference: https://rikiyat.github.io/blog/weight-space-programming/

## Implemented

- Exact 600 → 32 → 32 → 6 tanh architecture (20,486 parameters)
- Direct sparse weight writing, without gradients or training
- Deterministic toy Breakout evaluation (zero, follow, lead policies)
- 2,000 seeded episodes at 10% sticky actions; paired bootstrap confidence intervals
- Sticky-action sweep (0%, 10%, 25%, 50%) with frozen weights
- Five reproducible SVG visualizations from a committed JSON summary
- Optional MinAtar environment adapter; schema-validated JSON patch for future one-shot LLM edits
- Unit tests and GitHub Actions

## Reproduction limitations

This experiment uses **a simplified independent toy game** and **human-written weights**. It is not the original MinAtar harness and does not reproduce the paper's scores or demonstrate GPT-generated weights. The upstream MinAtar adapter is optional, and has not been used to obtain reported benchmark results. PPO scores cannot be compared without training/evaluating PPO in the same environment.

The main lead-controller achieves 8.017 mean reward at 10% sticky actions on 2,000 toy episodes, but it is sensitive to action noise (14.666 at 0% sticky; 2.25 at 50% sticky). Statistics and seeds are in results/benchmark-summary.json.

## Next steps

1. Verify the six-channel input and action mapping against the author's game wrapper, then evaluate on original MinAtar for 1,000 episodes per policy and 10,000 max steps.
2. Train a PPO baseline in that identical harness and archive checkpoints and training settings.
3. Run a genuinely **one-shot** LLM-produced sparse weight-patch experiment, saving its proposal before any rollouts or reward feedback.
4. Extend direct-weight circuits to Asterix and Seaquest, retaining exact action semantics.
5. Reproduce the paper's blind weight-reading task separately.
