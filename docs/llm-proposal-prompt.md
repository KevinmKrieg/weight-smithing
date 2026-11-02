# One-shot LLM weight-writing challenge

This is a **proposed experiment**, not a claim that an LLM has written working weights.

A language model must output JSON assigning a sparse subset of model parameters. Save the result as proposal.json and run:

~~~bash
python -m weightspace.patch --patch proposal.json --out artifacts/llm.npz
~~~

The patch validator accepts finite scalar assignments to known tensors only, with valid shapes and no duplicate indices. It NEVER runs model-generated code. Evaluation of arbitrary loaded models remains a future step.

## Prompt to give the LLM

You are programming a neural network directly in weight space with no training data, optimizer, gradients, environment access, code execution, or iterative feedback. Generate weight assignments only.

Game: Breakout on a 10x10 grid. A bottom-row paddle moves left/right to intercept a diagonally moving ball and earns rewards by breaking bricks. Missing the ball ends the episode. The current and previous ball positions can indicate direction.

Network: two tanh hidden layers. Equations: h1=tanh(w1 @ x+b1); h2=tanh(w2 @ h1+b2); logits=w3 @ h2+b3. Shapes: w1=(32,600), b1=(32,), w2=(32,32), b2=(32,), w3=(6,32), b3=(6,). All values initially zero.

Input: binary 6x10x10 grid, flattened channel-first; flat index = channel*100 + row*10 + column. Channels: paddle=0, ball=1, brick=2, obstacle=3, previous ball=4, gauge=5.

Actions: no-op=0, left=1, up=2, right=3, down=4, fire=5. Valid Breakout moves are 0, 1, 3.

Return ONLY a JSON object of this shape:
~~~json
{"edits":[{"tensor":"w1","index":[0,100],"value":0.1}]}
~~~

Each edit assigns one scalar; unspecified values remain zero. Do not repeat indices. Up to 20,486 edits allowed. Your task is to intercept the ball as often as possible, including reflections. Do not claim tested performance.

## Blind trial procedure

Save the unmodified model proposal first, then evaluate on a fixed seed list and compare to the hand-authored and random baselines. Do not incorporate reward feedback into the prompt or allow multiple proposals if describing the trial as one-shot.
