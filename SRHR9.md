# Gelion SRHR v9.0 — Unified Architecture
## Reconciling Grounded Coding-Agent Behavior with Novel Generation

**Status:** builds on v8.0. Layers 1, 2, 4 (core decoder), 6, and §9 (capacity analysis)
are structurally unchanged — referenced, not restated in full. This document specifies
what changes: §3.3 (memory is now dual-substrate), a new §3.4 (novelty gate — the load-
bearing addition), §5 (mode-dependent critique), §8 (mode-dependent reward). §10's
generalization analysis is amended, not superseded — the core finding stands, this
document narrows where the risk lives rather than removing it.

---

## 0. The actual conflict, stated precisely

Not "agent mode vs. exploration mode" as two disconnected systems — that was the wrong
frame. The real conflict was **one hardcoded assumption**: that retrieved context and
field-memory grooves should always be trusted and always accumulate. That assumption is
correct for a coding agent working inside a known codebase (reuse existing correct
patterns, get faster at recognizing them over time — entrenchment here is *skill*, not a
bug) and wrong for genuinely novel input (there's nothing correct to reuse yet, and
premature entrenchment forces new material through old grooves that don't fit it).

Fix: make trust in retrieval/memory a function of **how novel the current input actually
is**, computed from the system's own retrieval confidence — not declared by the user,
not a separate mode switch, a continuous gate. One architecture, one decoder, behavior
shifts automatically with what it's looking at.

---

## 1. §3.4 — Novelty Gate (new)

Compute a novelty score from retrieval confidence — no new subsystem, this reuses 3A's
own similarity metric:

$$\nu = 1 - \max_{\mathbf{A}_k \in \mathcal{R}} \text{cos\_sim}(\mathbf{q}, \mathbf{A}_k)$$

$\nu \to 0$: close precedent exists in the retrieval index. $\nu \to 1$: nothing similar
has been seen before. This single scalar (or a smoothed/windowed version of it, to avoid
jitter on borderline cases) drives four downstream gates:

**a) Conditioning fusion weight** — at the point where Layer 4 combines
$V_{context}$ (STHE), retrieved context (3A), and field-memory bias (3C), retrieval
influence is down-weighted as novelty rises:

$$w_{retrieval} = (1-\nu)\cdot w_{base}, \qquad w_{decoder\_prior} = 1-w_{retrieval}$$

This is the mechanism that stops the earlier "retrieval-paraphrase crutch" risk from
being uniform: when $\nu$ is low (a known pattern), leaning on retrieval is exactly
correct — reuse the known-good solution, don't reinvent it. When $\nu$ is high, the gate
itself forces the decoder to rely on its own trained distribution instead of stretching a
distant retrieval match to fit, which is what produces confident-but-wrong paraphrase.

**b) Sampling temperature:**

$$T = T_{base} + \alpha\cdot\nu$$

Wider sampling under genuine novelty — more of the decoder's learned distribution gets
explored instead of collapsing toward the single nearest retrieved pattern.

**c) Deliberation budget (Layer 4.5):**

$$N_{passes} = N_{base} + \beta\cdot\nu$$

Think longer specifically when there's no precedent to lean on — this is the direct fix
for "раздумывать над новым": more scratchpad iteration is spent exactly where it's
needed, not uniformly.

**d) Field-memory write routing** — see §2 below, this is where entrenchment gets
resolved rather than just flagged as a risk.

---

## 2. §3.3 revised — Dual-Substrate Field Memory

v8.0 flagged waveguide entrenchment (single field, all resonant patterns get carved in
permanently) as actively working against generalization. Fix: **two velocity traces,
not one**, both modulating the same physical grid, different lifetimes:

$$v_{eff}(\mathbf{r},t) = v_{base} + v_{stable}(\mathbf{r},t) + w_{explore}(t)\cdot v_{explore}(\mathbf{r},t)$$

- $v_{stable}$: the original NPG dynamics from §1.1, slow $\gamma$, long persistence.
  This is where accumulated **skill** lives — recognized codebase patterns, house style,
  APIs used repeatedly. Entrenchment here is desired: getting faster and more consistent
  on familiar ground is the point of a coding agent that's worked in a project a while.
- $v_{explore}$: same functional form, but reset per exploration episode (fast $\gamma$,
  short persistence by default) and **gated by novelty**: $w_{explore}(t)$ scales with
  the running average of $\nu$ over the current session/episode. High-novelty work
  reinforces $v_{explore}$, not $v_{stable}$ — a novel excursion doesn't immediately
  groove the substrate that biases all future behavior.

**Consolidation (promotion path):** an idea produced under high-$\nu$ exploration only
gets written into $v_{stable}$ (and, separately, only gets added to the retrieval index
3A) after it's **validated** — passes Layer 5 execution feedback for code, or passes the
critique pass's soundness check for non-code reasoning (§3 below). Unvalidated
exploration content stays in the fast-decaying $v_{explore}$ trace and fades on its own.
This is the actual mechanism that lets the same system both invent freely and become
skilled — invention happens in a substrate that doesn't yet influence default behavior;
only checked-out results get promoted to the substrate that does. (Loose analogy to
memory consolidation, stated as an analogy, not a claim about how biological memory
works.)

---

## 3. §5 revised — Mode-Dependent Critique

The Layer 4.5 critique pass now branches on what kind of claim is being checked, not on
a declared mode:

- **Code output:** unchanged from v8.0 — Layer 5 execution feedback (compile/test),
  first-block-extraction caveat from v8.0 §6 still applies.
- **Non-code reasoning (physics, design, novel algorithm proposals):** two separate,
  concrete checks, not a vague "does this seem good":
  - **Soundness:** internal consistency — does the conclusion follow from stated
    premises, dimensional/unit consistency, known-limit checks where applicable
    (unchanged from v8.0).
  - **Originality, now made concrete instead of aspirational:** embedding distance
    between the draft output and its nearest neighbor in the retrieval index. If this
    distance is *low* while $\nu$ (input novelty) was *high*, that's a specific,
    checkable signal that the decoder collapsed toward a retrieval match despite the
    gate telling it not to — flag for either re-sampling at higher $T$ or explicit
    human review, rather than silently shipping a paraphrase as if it were new work.

This makes "не копирует, а генерирует" a measurable property of a specific output, not
just an architectural hope — you can check, per response, whether output-vs-nearest-
retrieved distance is consistent with the novelty gate's own assessment of the input.

---

## 4. §8 revised — Mode-Dependent Reward

RL fine-tuning reward now weights by task type detected the same way as everywhere
else — via $\nu$ and whether Layer 5 or the critique pass produced the check:

$$R = \underbrace{w_{exec}\cdot R_{exec}}_{\text{code, always active when applicable}} + \underbrace{w_{sound}\cdot R_{soundness}}_{\text{always}} + \underbrace{w_{orig}(\nu)\cdot R_{originality}}_{\text{scales with input novelty}}$$

$w_{orig}(\nu)$ rises with $\nu$ — the training signal explicitly rewards originality
more when the input was genuinely novel, and de-emphasizes it (in favor of grounded
correctness/reuse) when a close precedent existed and reuse was the right call. This is
what prevents training from pushing the whole system toward either extreme uniformly —
it learns the same $\nu$-dependent behavior the inference-time gates enforce, so the
policy and the reward are aligned instead of the reward silently fighting the gating.

---

## 5. What this does and doesn't fix

**Fixes:** the specific structural conflict identified previously — entrenchment now
accumulates skill where skill is wanted (stable substrate, validated content only) and
stays out of the way of genuinely novel input (separate fast-decaying substrate,
down-weighted retrieval influence, wider sampling, more deliberation, explicit
originality check) — all driven by one measurable quantity ($\nu$) instead of a manual
mode switch, so a single coding session that suddenly hits a genuinely unprecedented
problem shifts behavior automatically, without the user having to declare "now be
creative."

**Doesn't fix:** §10's core finding from v8.0 — that STHE's structural prior is still
unvalidated, and that Layer 4's actual competence at either coding or novel reasoning
still depends entirely on training data/scale/curriculum, same as before. This document
resolves the *routing* conflict between the two use cases; it does not manufacture new
generalization capacity. The ablation protocol from v8.0 §10.6 is still the thing that
tells you whether any of this is earning its complexity versus a plainer conditioning
scheme — that hasn't changed and shouldn't be skipped because the routing problem got
solved.
