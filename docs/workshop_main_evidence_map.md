# `workshop_main.tex` evidence map

Scope: this note applies only to `workshop_main.tex`. Reports are derived views; check the named row-level file and manifest before promoting a number.

## Section 2.2.2 audit

- The old text incorrectly called the object probes general relative-direction queries. They ask whether the goal, key, or door is immediately adjacent in each cardinal direction.
- The canonical registry also contains four coordinate questions and action-conditioned successor-coordinate questions.
- Each action has three categorical consequence probes: wall collision, key possession after the move, and door-open status after the move.
- Reasoning-time work uses a subset: four walls, key possession, door status, and three consequence probes for the currently preferred action. The main scaled run uses candidate-token log probabilities at temperature 0.7, not repeated samples.

Sources: `src/reveng/experiments/behavioral_probe_questions.py`, `behavioral_probe_runner.py`, `reasoning_belief_action.py`, and `outputs/experiment2_behavioral_beliefs/gpt_oss_local_sentence_matched46_v1/run_report.md`.

## Evidence map

| Draft location | Existing evidence or gap | Repository source | Recommended use |
|---|---|---|---|
| Sec. 3.1 report beliefs | The clean matched wall table has 15 states per direction; the failure slice has 16 states and only three eligible wall hits. | `data/behavioral_probes/released_cognitive_alignment_{pre,post}/`; `case_studies_balanced_failure_modes/` | Keep as examples, not population estimates. Scale endpoint summaries on the matched-46 or full 1,276-decision corpus. |
| Sec. 3.2 value results | Current 3,194-step PRE/POST gap and 59% versus 99.5% IRL results are stronger when splits, held-out unit, uncertainty, and repeated-state handling are stated. | `report_branch_behav_probes.tex`; relevant value outputs indexed by `docs/artifact_index.csv` | Add trajectory/grid-held-out intervals. Do not interpret the planner-supervised fit as the model's implemented value function. |
| Sec. 3.3 failure decomposition | The text defines three failure classes but does not report their counts. | No completed joined classification found. | Join held-out probe beliefs, both value models, optimal-action sets, and emitted actions; bootstrap proportions by trajectory. |
| Sec. 4 belief trajectories | Observable beliefs improve three-sentence optimality-loss AUPRC from 0.273 to 0.334, change +0.061, 95% CI [+0.014,+0.123]; one-sentence evidence is inconclusive. | `outputs/hypothesis_tests/practical_action_event_monitor_v1/prediction_report.md` | Useful prospective evidence; call it short-horizon association, not causal prediction. |
| Sec. 4 commitment/change points | Offline change points are near recommendation changes in 86.9% of cases versus 65.3% for progress-matched sentences. | `outputs/reader_facing/belief_action_reasoning_summary_v1/EVIDENCE_AUDIT.md`; CPD reports | Supports intervention targeting, not a semantic or causal boundary. |
| Sec. 4.3/5 target selection | Seven of 18 exact optimality losses at primary change points have all nine categorical reports correct; 66/161 broader losses preserve six correct state reports across the boundary. | `SEMANTIC_INTERVENTION_CANDIDATES.md`; `EVIDENCE_AUDIT.md` in the reader-facing summary | Strong candidates for sentence deletion/replacement and belief-clause tests. |
| Sec. 4.4 semantics | No exact action-event association survives correction; route deliberation has higher belief entropy. | `SEMANTIC_CROSS_ANALYSIS_SYNTHESIS.md` | Use to rule out a single verbal “correction” mechanism and stratify interventions. |
| Sec. 5 whole-trace transplant | Internal paper summaries report 100% donor-action following and a 76% flip rate, but row-level records, N, matching, and uncertainty were not found locally. | `belief_action_gap.tex`; reader-facing synthesis | Provisional only. Restore raw rows before a quantitative claim. |
| Sec. 5 belief replacement | Eight traces: grid-only differs from full CoT in 6/8; six reports differ in 5/8; rescue 1/6; mean TV 0.731 versus 0.750. | `outputs/hypothesis_tests/belief_replacement_action_pilot_v1/` | Fits as a behavioral compression/sufficiency baseline, not causal mediation. |
| Sec. 5 activation patching | The 50-pair artifact substitutes saved layer-15 vectors without downstream recomputation; 26/50 are disruptive. | `data/cf/eval_results/aggregate_summary.json`; `counterfactual_activation_patching.py` | Do not cite as live activation patching or evidence of no causal effect. |

## Priority experiments

1. Restore and audit the whole-trace transplant rows. Report N, donor-pair construction, action-disagreement eligibility, exact binomial intervals, and action-distribution shifts.

2. Scale belief replacement beyond eight traces, stratifying by final action, success, report correctness, and the strict 7-case/broader 66-case optimality-loss sets. Add repeated seeds and trajectory-clustered intervals.

3. Run the trusted-verifier clause experiment with grid-only, model-report, verified-correct, minimally counterfactual, and irrelevant-clause controls. Never include an action recommendation in a belief clause.

4. On the selected loss sentences, compare original, deletion, truthful rewrite, one-fact counterfactual rewrite, and length-matched unrelated replacement. Re-read both the action distribution and beliefs after each edit; cross pre- versus post-commitment positions.

5. Run true forward-pass activation patching across layers and positions with donor-aligned, same-state, random-position, and norm-matched controls. The existing saved-trace baseline is not a substitute.

## Evidence rules

- Split and bootstrap by trajectory or grid, not by sentence/readout row.
- Treat behavioral reports as elicited information, not the latent belief used in the original forward pass.
- Treat retrospective commitment and BEAST points as selection devices unless intervened on.


## 2026-08-31 matched-46 belief-clause result

Artifact: outputs/hypothesis_tests/belief_clause_intervention_matched46_v1/.
Implementation: src/reveng/experiments/belief_clause_intervention.py; runner:
scripts/run_belief_clause_intervention.py.

- Design: 46 decisions from 31 trajectories. Each no-CoT query retains the grid and
  key status. Conditions are model reports, six simulator-verified facts, one
  deterministically flipped wall fact, and six irrelevant formatting facts. The
  existing position-zero action distribution is the grid-only reference; the final
  prefix distribution is the full-CoT reference.
- Full CoT is optimal in 38/46 cases. Grid-only is optimal in 17/46 and differs from
  full CoT in 34/46.
- Model-report replacement is optimal in 23/46 and differs from full CoT in 31/46.
  The optimal-rate change against grid-only is +0.130 with a trajectory-clustered
  95% interval of [-0.043, +0.314].
- Verified-truth replacement is optimal in 11/46 and differs from full CoT in 36/46.
  It corrects 2 suboptimal full-CoT actions and degrades 29 optimal full-CoT actions.
- The one-false-wall condition is optimal in 20/46; the irrelevant control is
  optimal in 22/46. Both outperform verified truth in paired clustered contrasts.
  Model reports and the irrelevant control differ by only +0.022 in optimal rate,
  95% interval [-0.143, +0.186], although their top actions differ in 21/46 cases.
- The all-six-reports-correct subgroup is small (12 decisions) and performs poorly
  under every clause replacement. Treat this as exploratory selection structure,
  not evidence that correct reports are harmful.

Interpretation: the scaled result does not support the six local reports as a
sufficient replacement for CoT, and it does not show a truth-specific verifier
benefit. Strong prompt-induced action shifts and the irrelevant-control result make
surface form and clause order live confounds. Keep the eight-trace pilot only as
historical provenance. Before adding this result to the paper, counterbalance clause
order and wording, add route/goal-direction and action-value clauses, and predefine
the expected action under each counterfactual believed state.
