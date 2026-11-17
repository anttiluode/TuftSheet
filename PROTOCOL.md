# Protocol: TuftSheet

Written 17 November 2026 by Claude Opus 5.5, **before** the batch. Committed and pushed before `experiments/run.py` is run; the receipt holds this file's hash.

## The question

The machines of [ApicalRuleReversalMachine](https://github.com/anttiluode/ApicalRuleReversalMachine) and [WaveCortex](https://github.com/anttiluode/WaveCortex) held the world's context in **one** apical state *g*, shared by every tuft. One number can hold only one context.

Antti's picture is different. It has many local machines, each relating its local input to the state just above it, with the context moving horizontally through Layer 1. **The world is then the agreement pattern of a sheet, not a variable.**

Can such a sheet:
- hold two independent contexts at once, in two places, where one *g* cannot;
- share a surprise sideways, so a column that has not yet been surprised gets the new context;
- keep the boundary, so a switch on one side does not drag the other side along;
- carry evidence over distance in time, so nearer columns get it first;
- learn where the boundary is?

## The machine

`sheet/sheet.py`. 24 relational columns in a row, in the scalar form that WaveCortex found more accurate than the wave form. Each column predicts

```math
\text{prediction}_i = A\,\big(W_i + g_i V_i\big), \qquad g_i = \tanh(\phi_i),
```

where *φ* is a **sheet**: a field of log-odds of the rule over the row.

- **Ping.** After a trial at column *i*, its surprise is pushed into the sheet at its own place, by ARM's normalized step (local now): $`\phi_i \mathrel{-}= \eta\, r_i A V_i / (V_i^2+\kappa)`$.
- **Spread.** Between trials, the sheet diffuses along its links *c* and leaks slowly toward 0, so old evidence fades.
- **Weights.** W and V learn by ARM's three-factor rule during training only.

Variants, all trained on the same sequences:
- `uniform`: all links 1;
- `learned`: links learned from sign agreement of neighbours' recent pings;
- `oracle`: the true boundary link cut (an upper bound for a sheet);
- `local`: no links, so every column is alone;
- `global`: one shared *g*, which is ARM's machine.

**The ideal observer** knows the regions, the hazard and the noise: one exact two-state filter per region.

**What the sheet is, plainly.** It is a diffusive field: a sheet with neighbour links. It is **not** WaveComputer's linear wave field. A wave passes on and does not hold state; a context has to stay. So this tests whether a *field* earns its place, not whether *waves* do.

## The world

- Columns 0–11 form the left region, 12–23 the right. Each region has a hidden rule ±1, which flips by itself with probability 1/30 per trial.
- Each trial cues one column at random with A = ±1. The outcome is B = rule(region)·A, flipped with probability 0.1.
- Training is 4000 trials; the test is 600 trials, with weights and links frozen and the sheet still running.

## Development, disclosed

Seeds 981–986 only.
1. **Blocks of 20–40 trials made the two rules anti-correlated for long stretches.** One global *g* exploited that with opposite gated-weight signs per region. I switched to a constant hazard, so the regions are independent.
2. **Slow sideways spread** (about 1 column per trial) let columns learn opposite sign conventions, so the sheet cancelled itself (accuracy about 0.55). Faster spread fixed it: D = 10, 25 ticks of 0.04 per trial.
3. **Push strength η:** 1, 3 and 8 were tried; 8 was chosen.
4. **Learned links:**
   - with fast traces (0.9) they put the weakest link anywhere;
   - with slow traces (0.98, step 0.01) the boundary link comes out below the typical link in most bodies, but never as the weakest.

   Learned links stayed worse than uniform links in 6/6 bodies.

The final defaults are in `sheet/sheet.py`; the development bodies are in [`experiments/dev_results.json`](experiments/dev_results.json).

| Development medians | uniform | learned | oracle | local | global | ideal |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.80 | 0.76 | 0.83 | 0.61 | 0.68 | 0.86 |

## Decision rules (seeds 600001–600032; "≥ 29" is 90%, "≥ 24" is 75%)

**Sign accuracy** is read along A: the right sign is the current rule of the column's region.

- **T1. Two contexts at once.** The uniform sheet's accuracy has median ≥ 0.75, and it beats `global` in ≥ 29 bodies and `local` in ≥ 29 bodies.
- **T2. Sideways transfer.** On a column's first trial after its region switched, when a column within distance 3 had already been surprised since the switch: uniform sheet median ≥ 0.70, `local` median ≤ 0.50.
- **T3. The boundary holds.** On the other region's trials in the 10 trials after a switch (with its own rule unchanged): uniform sheet median ≥ 0.85, and above `global` in ≥ 24 bodies.
- **T4. Evidence travels.** The uniform sheet loses more transfer accuracy with distance (near ≤ 3 minus far ≥ 6) than the ideal observer does. The median of the per-body difference must be > 0.05.
- **T5. The cut.** With *g* silenced, the uniform sheet's accuracy has median between 0.40 and 0.60.
- **T6. The boundary is learned (open).** The learned boundary link is below the median of the other links in ≥ 24 bodies.

**Reported without a rule:**
- learned against uniform;
- the oracle bound;
- squared error to the ideal;
- the boundary link's rank;
- whether it is the weakest link.

## Predictions, recorded before the run

- **T1–T5 hold,** near the development numbers.
- **T6 is a coin toss.** Development gave 4 of 6, under the 75% bar. **The weakest link will almost never be at the boundary.**
- **Learned links will be worse than uniform links in most bodies.** A boundary-learning rule that only listens to rare pings is not yet good enough.
- **The oracle beats uniform only a little** (development: 0.83 against 0.80). A sheet does not need to know the boundary to hold two contexts when evidence keeps arriving on both sides.
